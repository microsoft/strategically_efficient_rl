import sys  # This is never used

import ray
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.tune import Trainable
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from ray.tune.utils import merge_dicts
from ray.tune.logger import NoopLogger

from algorithms.self_play.evaluation import make_checkpoint_config, make_random_config, \
        build_wrapper_cls, checkpoint_eval, random_eval, build_eval_function


SIMULTANEOUS_CONFIG = {
    # The RLLib Trainable to run for each agent
    "alg": "PPO",
    # Number of evaluation episodes against a uniform random policy
    "multiagent_eval_episodes": 20,
    # Number of training iterations between random evaluations
    "multiagent_eval_interval": 1,
    # Number of workers for multi-agent evaluation
    "multiagent_eval_num_workers": 0,
    # Evaluate against a random policy
    "random_eval": True,
    # Population of policies against which we should evaluate
    "population": [],
}

class Simultaneous(Trainable):

    def _setup(self, config):
        config = merge_dicts(SIMULTANEOUS_CONFIG, config)
        config = merge_dicts(COMMON_CONFIG, config)

        # Build an instance of the environment - needed to define the multiagent config
        env_name = config.get("env")
        env_config = config.get("env_config", {})

        if _global_registry.contains(ENV_CREATOR, env_name):
            env_creator = _global_registry.get(ENV_CREATOR, env_name)
        else:
            import gym
            env_creator = lambda env_config: gym.make(env_name)

        env = env_creator(env_config)

        # Build multi-agent configs for training and evaluation
        policies = dict()
        for pid in env.observation_space_dict.keys():
            policies[f"learned_policy_{pid}"] = (
                    None,
                    env.observation_space_dict[pid],
                    env.action_space_dict[pid],
                    {}
                )
        
        policies_to_train = list(policies.keys())

        # Get config for population based evaluation
        checkpoints = config.pop("population")
        policies.update(make_checkpoint_config(checkpoints))  # We use the checkpoints to define the multi-agent configs

        # Add random policies to evaluation config if needed
        random_eval = config.pop("random_eval")

        if random_eval:
            policies.update(make_random_config(env.observation_space_dict, \
                    env.action_space_dict))

        # Define multiagent configs for evaluation and training
        config["multiagent"] = {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": lambda idx: policies_to_train[idx],
        }

        config["evaluation_config"] = {  # This config is used for evaluation - it doesn't define the evaluation process on its own, just defines the workers
            "multiagent": {
                "policies": policies,
                "policies_to_train": [],
                "policy_mapping_fn": lambda pid: pid,
            }
        }

        # Define evaluation config  # These just define the computational constraints of evluation
        config["evaluation_num_workers"] = config.pop("multiagent_eval_num_workers")
        config["evaluation_interval"] = config.pop("multiagent_eval_interval")

        # Build custom evaluation function  # For exact evaluation in openspiel, we probably don't need thsi
        base_mapping = {idx: policy for idx, policy in enumerate(policies_to_train)}
        eval_episodes = config.pop("multiagent_eval_episodes")
        config["custom_eval_function"] = build_eval_function(checkpoints, base_mapping, eval_episodes, random_eval)

        # Build the base Trainable  # Here we actually build the trainable class - this has a bug which cuases it to create a separate results directory
        trainer_cls = get_trainable_cls(config.pop("alg"))
        self._trainer = trainer_cls(config, 
                            env=build_wrapper_cls(env_creator), 
                            logger_creator=lambda config: NoopLogger(config, self._logdir))

        # Copy checkpoint weights to evaluation policies  # This is where we pull weights from previous checkpoints
        checkpoint_weights = dict()
        for checkpoint in checkpoints:
            checkpoint_weights[checkpoint.policy_id] = ray.get(checkpoint.weights)

        # TODO: May want to look at the custom scheduler
        self._trainer.workers.local_worker().set_weights(checkpoint_weights)  # This part is tricky, how do we do load weights while still using the standard run-experiment pipeline?

    def train(self):  # To do side-training, we might just want to run two Trainers separately
        return self._trainer.train()  # Here we just execute the underlying trainable - all the Simultaneous Wrapper does is manage evaluation

    def _stop(self):
        self._trainer._stop()

    def _save(self, tmp_checkpoint_dir):
        return self._trainer._save(tmp_checkpoint_dir)

    def _restore(self, checkpoint):
        self._trainer._restore(checkpoint)
