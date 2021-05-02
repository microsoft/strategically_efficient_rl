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

from algorithms.agents.deep_nash_v1.deep_nash_policy_v1 import DeepNashPolicy


DEFAULT_CONFIG = with_common_config({
    # Replay buffer size (keep this fixed for now, we don't have a better way of setting it)
    "buffer_size": 100000,
    # Update batch size (number of transition to pass for each batch)
    "batch_size": 128,
    # Ratio between the number of steps sampled, and the number of updated performed
    "sampling_ratio": 5,
    # Learning rate for Adam updates
    "learning_rate": 0.0001,
    # Discount factor
    "gamma": 0.99,
    # Learning rate for multiplicative weights
    "beta": 0.05,
    # Implicit exploration constant
    "implicit_exploration": 0.05,
    # Whether the policy and Q-function should share feature layers
    "share_layers": False,
    # Q-network hidden layers
    "hidden_sizes": [256, 256],
    # TensorFlow activation function
    "activation": "tanh",
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
    _name = "DEEP_NASH_V1"
    _default_config = DEFAULT_CONFIG

    def __init__(self, config=None, env=None, logger_creator=None):
        Trainer.__init__(self, config, env, logger_creator)

    @override(Trainer)
    def _make_workers(self, env_creator, policy, config, num_workers):
        return Trainer._make_workers(self, env_creator, policy, config, num_workers)
    
    @override(Trainer)
    def _init(self, config, env_creator):

        # Define rollout-workers
        self.workers = self._make_workers(env_creator, self._policy, config, self.config["num_workers"])

        # Define replay buffers dictionary
        self._replay_buffers = {}

        # Compute effective planning horizon
        self._horizon = np.log(0.1) / np.log(config["gamma"])

        # Initialize learning rate and sample count
        self._num_steps_observed = 0

    def _train(self):

        # Synchronize weights across remote workers if necessary
        if self.workers.remote_workers():
            weights = ray.put(self.workers.local_worker().get_weights())

            for worker in self.workers.remote_workers():
                worker.set_weights.remote(weights)

        # Generate samples and add them to the replay buffer
        if self.workers.remote_workers():
            samples = SampleBatch.concat_samples(
                ray.get([
                    worker.sample.remote() 
                    for worker in self.workers.remote_workers()
                ]))
        else:
            samples = self.workers.local_worker().sample()
        
        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({
                DEFAULT_POLICY_ID: samples
            }, samples.count)

        self._num_steps_observed += samples.count
    
        # Accumulate episode results
        episodes, _ = collect_episodes(self.workers.local_worker(), self.workers.remote_workers())
        results = summarize_episodes(episodes)
        results[TIMESTEPS_THIS_ITER] = samples.count

        # Add data to replay buffers and do updates of policies
        policy_fetches = {}

        for policy_id, policy_batch in samples.policy_batches.items():
            if policy_id in self.workers.local_worker().policies_to_train:
                if policy_id not in self._replay_buffers:
                    self._replay_buffers[policy_id] = ReplayBuffer(self.config["buffer_size"])

                for sample in policy_batch.rows():
                    self._replay_buffers[policy_id].add(sample)

                required_updates = math.ceil(policy_batch.count / self.config["sampling_ratio"])
                average_fetches = defaultdict(list)

                for _ in range(required_updates):
                    batch = self._replay_buffers[policy_id].sample_batch(self.config["batch_size"])
                    fetches = self.workers.local_worker().policy_map[policy_id].learn_on_batch(batch)

                    for key, value in fetches["learner_stats"].items():
                        if value is not None and not isinstance(value, dict):
                            average_fetches[key].append(value)

                for key, value in average_fetches.items():                    
                    average_fetches[key] = np.mean(value)
                
                policy_fetches[policy_id] = average_fetches

        results["learner/info"] = policy_fetches

        # Update learning rate across workers
        learning_rate = self.config["beta"] * np.sqrt(self._num_steps_observed / self._horizon)
        results["learner/info/learning_rate"] = learning_rate
        self.workers.foreach_trainable_policy(lambda policy, pid: policy.set_learning_rate(learning_rate))

        return results
