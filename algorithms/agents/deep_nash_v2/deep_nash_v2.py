from collections import defaultdict
import math
import numpy as np

import ray
from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.agents import with_common_config
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.tune.result import TIMESTEPS_THIS_ITER

from algorithms.agents.deep_nash_v2.deep_nash_policy_v2 import DeepNashPolicy, CURRENT_ACTION_MASK


DEFAULT_CONFIG = with_common_config({
    # The GAE(lambda) parameter.
    "lambda": 0.95,
    # The intrinsic GAE(lambda) parameter.
    "intrinsic_lambda": 0.95,
    # The intrinsic reward discount factor parameter.
    "intrinsic_gamma": 0.95,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 200,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # Stepsize of SGD.
    "lr": 0.001,
    # Learning rate schedule.
    "lr_schedule": None,
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Use a separate head for intrinsic value function?
    "intrinsic_head": True,
    # The number of agents in the environment - for joint curiosity
    "num_agents": 1,
    # Size of the replay buffer used to compute the time averaged policy
    "average_buffer_size": 100000,
    # Size of minibatches for the cloning loss
    "average_minibatch_size": 128,
})


class ReplayBuffer:

    def __init__(self, initial_size):
        self._max_size = initial_size
        self._index = 0
        self._data = []

    def expand(self, new_size):
        if new_size < self._max_size:
            raise ValueError("New buffer size is smaller than current size - cannot shrink buffer")

        self._max_size = new_size

    def add(self, sample):
        if len(self._data) <= self._index:
            self._data.append(sample)
        else:
            self._data[self._index] = sample
        
        self._index = (self._index + 1) % self._max_size

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, len(self._data), batch_size)
        batch = defaultdict(list)

        for idx in indices:
            for key, value in self._data[idx].items():
                batch[key].append(value)
        
        for key in batch.keys():
            batch[key] = np.stack(batch[key])

        return SampleBatch(batch)


class DeepNash(Trainer):

    _policy = DeepNashPolicy
    _name = "DEEP_NASH_V2"
    _default_config = DEFAULT_CONFIG

    def __init__(self, config=None, env=None, logger_creator=None):
        Trainer.__init__(self, config, env, logger_creator)

    @override(Trainer)
    def _make_workers(self, env_creator, policy, config, num_workers):
        return Trainer._make_workers(self, env_creator, policy, config, num_workers)
    
    @override(Trainer)
    def _init(self, config, env_creator):

        # Define rollout-workers - where do we get the policy and default config from?
        self.workers = self._make_workers(env_creator, self._policy, config, self.config["num_workers"])

        # Define replay buffers dictionary
        self._replay_buffers = {}

    def _build_batch(self, samples, average_batch):
        indices = np.random.randint(0, samples.count, self.config["sgd_minibatch_size"])
        batch = defaultdict(list)

        for key, value in samples.items():
            batch[key] = value[indices]
        
        for key in batch.keys():
            batch[key] = np.stack(batch[key])

        batch = SampleBatch(batch)

        # batch[CURRENT_ACTION_MASK] = np.ones(batch.count)
        average_batch[CURRENT_ACTION_MASK] = np.zeros(average_batch.count)

        return SampleBatch.concat_samples([batch, average_batch])

    def _train(self):

        # Synchronize weights across remote workers if necessary
        if self.workers.remote_workers():
            weights = ray.put(self.workers.local_worker().get_weights())

            for worker in self.workers.remote_workers():
                worker.set_weights.remote(weights)

        # Generate samples and add them to the replay buffer
        samples = []
        while sum(batch.count for batch in samples) < self.config["train_batch_size"]:
            if self.workers.remote_workers():
                samples.append(SampleBatch.concat_samples(
                    ray.get([
                        worker.sample.remote() 
                        for worker in self.workers.remote_workers()
                    ])))
            else:
                samples.append(self.workers.local_worker().sample())
        
        samples = SampleBatch.concat_samples(samples)

        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({
                DEFAULT_POLICY_ID: samples
            }, samples.count)
        
        # Accumulate episode results
        episodes, _ = collect_episodes(self.workers.local_worker(), self.workers.remote_workers())
        results = summarize_episodes(episodes)
        results[TIMESTEPS_THIS_ITER] = samples.count

        # Add data to replay buffers and do updates of policies
        policy_fetches = {}

        for policy_id, policy_batch in samples.policy_batches.items():
            if policy_id in self.workers.local_worker().policies_to_train:
                if policy_id not in self._replay_buffers:
                    self._replay_buffers[policy_id] = ReplayBuffer(self.config["average_buffer_size"])

                for sample in policy_batch.rows():
                    self._replay_buffers[policy_id].add(sample)

                average_fetches = defaultdict(list)

                for _ in range(self.config["num_sgd_iter"]):
                    average_batch = self._replay_buffers[policy_id].sample_batch(self.config["average_minibatch_size"])
                    batch = self._build_batch(policy_batch, average_batch)

                    # TODO: This is where learning actually happens - how do we input two different batches
                    fetches = self.workers.local_worker().policy_map[policy_id].learn_on_batch(batch)

                    for key, value in fetches["learner_stats"].items():
                        if value is not None and not isinstance(value, dict):
                            average_fetches[key].append(value)

                for key, value in average_fetches.items():                    
                    average_fetches[key] = np.mean(value)
                
                policy_fetches[policy_id] = average_fetches

        results["learner/info"] = policy_fetches

        return results
