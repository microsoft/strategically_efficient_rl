"""Utiltity methods for multi-agent evaluation"""

from collections import defaultdict, namedtuple
import numpy as np
import pickle
import os

import ray
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import collect_metrics
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import merge_dicts

Checkpoint = namedtuple("Checkpoint", ["weights", "policy_cls", "observation_space",
                            "action_space", "config", "agent_id", "policy_id"])


class RandomPolicy(Policy):
    """ Fixed policy which samples actions uniformly at random. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {} # Not clear what the additional returns are?

    def get_weights(self):
        """ Needed to if we are using remote workers. """
        return None

    def set_weights(self, weights):
        """ Needed to if we are using remote workers. """
        pass

# Why are there two copies of this?
def build_wrapper_cls(env_creator):
    
    class EvalWrapper(MultiAgentEnv):
        """
        This class wraps an existing multi-agent env, allowing us to
        rewrite the agent ids passed to the policy_mapping_function
        """

        def __init__(self, config):
            self._env = env_creator(config)
            self._forward_mapping = {pid:pid for pid in self._env.action_space_dict.keys()}
            self._reverse_mapping = self._forward_mapping.copy()

            # Add mapping for "__all__" for "done" values
            self._forward_mapping["__all__"] = "__all__"

            # Mark the reverse mapping as valid
            self._mapping_updated = False
            
        def _forward(self, env_dict):
            return {self._forward_mapping[key]: value for key, value in env_dict.items()}

        def _reverse(self, action_dict):
            action_dict = {self._reverse_mapping[key]: value for key, value in action_dict.items()}
            self._update_reverse_mapping()  # Update reverse mapping if needed

            return action_dict

        def _update_reverse_mapping(self):
            if self._mapping_updated:           
                for key, value in self._forward_mapping.items():
                    self._reverse_mapping[value] = key
                
                self._mapping_updated = False

        def update_mapping(self, mapping_dict):
            for pid in self._env.action_space_dict.keys():
                self._forward_mapping[pid] = mapping_dict[pid] if pid in mapping_dict else pid   

            self._mapping_updated = True

        def reset(self):
            self._update_reverse_mapping()  # Otherwise, the mappings will be out of sync at the next step
            return self._forward(self._env.reset())

        def step(self, action_dict):
            obs, rew, done, info = self._env.step(self._reverse(action_dict))
            return self._forward(obs), self._forward(rew), self._forward(done), self._forward(info)
    
    return EvalWrapper


def _build_wrapper_cls(env_creator, base_mapping):

    class EvalWrapper(MultiAgentEnv):
        """
        This class wraps an existing multi-agent env, allowing us to
        rewrite the agent ids passed to 'policy_mapping_fn'.
        """

        def __init__(self, config):
            self._env = env_creator(config)
            self._forward_mapping = base_mapping.copy()
            self._reverse_mapping = {}

            for key, value in self._forward_mapping.items():
                    self._reverse_mapping[value] = key

            # Add mapping for "__all__" for "done" values
            self._forward_mapping["__all__"] = "__all__"

            # Mark the reverse mapping as valid
            self._mapping_updated = False

        def _update_reverse_mapping(self):
            if self._mapping_updated:           
                for key, value in self._forward_mapping.items():
                    self._reverse_mapping[value] = key
                
                self._mapping_updated = False

        def _forward(self, env_dict):
            return {self._forward_mapping[key]: value for key, value in env_dict.items()}

        def _reverse(self, action_dict):
            action_dict = {self._reverse_mapping[key]: value for key, value in action_dict.items()}
            self._update_reverse_mapping()  # Update reverse mapping if needed

            return action_dict

        def update_mapping(self, mapping_dict):
            for pid in self._env.action_space_dict.keys():
                self._forward_mapping[pid] = mapping_dict[pid] if pid in mapping_dict else pid   

            self._mapping_updated = True

        def reset(self):
            self._update_reverse_mapping()  # Otherwise, the mappings will be out of sync at the next step
            return self._forward(self._env.reset())

        def step(self, action_dict):
            obs, rew, done, info = self._env.step(self._reverse(action_dict))
            return self._forward(obs), self._forward(rew), self._forward(done), self._forward(info)
    
    return EvalWrapper


def update_mapping(eval_workers, mapping_dict):
    """ Update the mapping from agents to polices for all eval workers. """
    eval_workers.local_worker().foreach_env(lambda env: env.update_mapping(mapping_dict))

    for worker in eval_workers.remote_workers():
        worker.foreach_env.remote(lambda env: env.update_mapping(mapping_dict))


def evaluate(eval_workers, num_episodes):
    """ Run a set of evaluation episodes with the current policy mapping """
    num_workers = len(eval_workers.remote_workers())

    if 0 == num_workers:
        for _ in range(num_episodes):
            eval_workers.local_worker().sample()
    else:
        num_rounds = int(math.ceil(num_episodes / num_workers))
        num_episodes = num_rounds * num_workers

        for _ in range(num_rounds):
            ray.get([
                w.sample.remote()
                for w in eval_workers.remote_workers()
            ])

    return collect_metrics(eval_workers.local_worker(), eval_workers.remote_workers())


def random_eval(eval_workers, base_mapping, num_episodes):
    """ Evaluates each learned policy against a random agent. """
    agent_ids = frozenset(base_mapping.keys())
    mean_returns = defaultdict(list)
    min_return = defaultdict(lambda: np.infty)
    max_return = defaultdict(lambda: -np.infty)

    for agent_id in agent_ids:
        mapping = base_mapping.copy()
        mapping[agent_id] = f"random_policy_{agent_id}"
        update_mapping(eval_workers, mapping)
        metrics = evaluate(eval_workers, num_episodes)

        for pid in base_mapping.values():
            if pid in metrics["policy_reward_mean"]:
                mean_returns[pid].append(metrics["policy_reward_mean"][pid])
                min_return[pid] = min(min_return[pid], metrics["policy_reward_min"][pid])
                max_return[pid] = max(max_return[pid], metrics["policy_reward_max"][pid])

    metrics = defaultdict(dict)
    for pid in base_mapping.values():
        if pid in mean_returns:
            metrics["random_mean"][pid] = np.mean(mean_returns[pid])
            metrics["random_min"][pid] = min_return[pid]
            metrics["random_max"][pid] = max_return[pid]

    return metrics


def checkpoint_eval(eval_workers, checkpoints, base_mapping, num_episodes):
    mean_returns = defaultdict(list)
    min_return = defaultdict(lambda: np.infty)
    max_return = defaultdict(lambda: -np.infty)

    for checkpoint in checkpoints:
        mapping = base_mapping.copy()
        mapping[checkpoint.agent_id] = checkpoint.policy_id
        update_mapping(eval_workers, mapping)
        metrics = evaluate(eval_workers, num_episodes)

        for pid in base_mapping.values():
            if pid in metrics["policy_reward_mean"]:
                mean_returns[pid].append(metrics["policy_reward_mean"][pid])
                min_return[pid] = min(min_return[pid], metrics["policy_reward_min"][pid])
                max_return[pid] = max(max_return[pid], metrics["policy_reward_max"][pid])

    metrics = defaultdict(dict)
    for pid in base_mapping.values():
        if pid in mean_returns:
            metrics["population_mean"][pid] = np.mean(mean_returns[pid])
            metrics["population_min"][pid] = np.min(mean_returns[pid])
            metrics["population_max"][pid] = np.max(mean_returns[pid])
            metrics["absolute_min"][pid] = min_return[pid]
            metrics["absolute_max"][pid] = max_return[pid]

    return metrics


def build_eval_function(checkpoints, base_mapping, eval_episodes, is_random_eval):

    def eval_function(trainer, eval_workers):
        results = {}

        # Do random evaluation if needed
        if is_random_eval:
            results["random"] = random_eval(eval_workers, base_mapping, eval_episodes)

        # Do population evaluation if needed
        if len(checkpoints) > 0:
            results["population"] = checkpoint_eval(eval_workers, \
                    checkpoints, base_mapping, eval_episodes)

        return results
    
    return eval_function


def make_checkpoint_config(checkpoints):
    policies = dict()
    for checkpoint in checkpoints:
        policies[checkpoint.policy_id] = (
                checkpoint.policy_cls,
                checkpoint.observation_space,
                checkpoint.action_space,
                checkpoint.config
            )
    return policies

def make_random_config(obs_space_dict, action_space_dict):
    policies = dict()
    for pid in obs_space_dict.keys():
        policies[f"random_policy_{pid}"] = (
            RandomPolicy,
            obs_space_dict[pid],
            action_space_dict[pid],
            {}
        )
    return policies


def load_checkpoints(checkpoint_paths):
    checkpoints = []
    run_index = -1

    for checkpoint_index, path in enumerate(checkpoint_paths):
        run_index = -1

        # Load checkpoints from individual runs
        for run in os.listdir(path["path"]):
            run = os.path.join(path["path"], run)

            if os.path.isdir(run):
                run_index += 1

                # Get parameters
                with open(os.path.join(run, "params.pkl"), "rb") as f:
                    params = pickle.load(f)

                # Modify config to remove any remote workers saved in the checkpoint
                params["num_workers"] = 0
                params["evaluation_num_workers"] = 0

                # Remove any unnecessary configuration parameters
                params.pop("alg", None)
                params.pop("symmetric", None)
                params.pop("self_play_round_stop", None)
                params.pop("self_play_pretrain_stop", None)

                # Get latest checkpoint - why are there two of these?
                last_save = sorted([save for save in os.listdir(run) if save.startswith("checkpoint_")])[-1]  # This may be broken
                last_save = os.path.join(run, last_save)

                checkpoint = [save for save in os.listdir(last_save) if (save.startswith("checkpoint-") and not save.endswith("metadata"))][0]
                checkpoint = os.path.join(last_save, checkpoint)

                # Build trainer and restore checkpoint
                trainer_cls = get_trainable_cls(path["alg"])
                trainer = trainer_cls(config=params)
                trainer.restore(checkpoint)

                # Extract policies
                policy_weights = trainer._trainer.get_weights()

                # print("\n\n===== POLICIES =====\n")
                #
                # for key, value in policy_weights.items():
                #     print(f"{key} - type {type(value)}")
                #
                # print("\n\n\n")

                for agent_id, policy_id in path["mapping"]:
                    policy_definition = trainer._trainer.config["multiagent"]["policies"][policy_id] # Why are we accessing the inner trainer?
                    policy = trainer._trainer.workers.local_worker().get_policy(policy_id)
                    policy_cls = policy_definition[0] or trainer._trainer._policy
                    observation_space = policy_definition[1]
                    action_space = policy_definition[2]
                    config = merge_dicts(trainer._trainer.config, policy_definition[3])

                    # Rewrite policy ID
                    new_policy_id = f"checkpoint_{checkpoint_index}_{run_index}_{agent_id}_{policy_id}"

                    # Rewrite weight keys - why do we do this?
                    weights = dict()
                    for key, value in policy_weights[policy_id].items():
                        new_key = [new_policy_id]
                        new_key.extend(key.split("/")[1:])
                        weights["/".join(new_key)] = value

                    # Store weights
                    weights = ray.put(weights)

                    # Create policy tuple - how are these loaded after the fact? - seems to use the multiagent config
                    checkpoints.append(Checkpoint(
                        weights=weights,
                        policy_cls=policy_cls,
                        observation_space=observation_space,
                        action_space=action_space,
                        config=config,
                        agent_id=agent_id,  # What is the difference between these?
                        policy_id=new_policy_id
                    ))

                # Kill Trainer to free any resources it may have loaded
                trainer.stop()

    return checkpoints
