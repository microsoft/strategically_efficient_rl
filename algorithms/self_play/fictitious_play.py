import cloudpickle
import copy
from itertools import cycle
import numpy as np
import os

import ray
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from ray.tune import Trainable
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.tune.utils import merge_dicts
from ray.tune.logger import NoopLogger
from ray.tune.result import (TIME_THIS_ITER_S, TIME_TOTAL_S,
                             TIMESTEPS_THIS_ITER, TIMESTEPS_TOTAL,
                             EPISODES_THIS_ITER, EPISODES_TOTAL,
                             TRAINING_ITERATION)

from algorithms.self_play.evaluation import make_checkpoint_config, make_random_config, \
        build_wrapper_cls, checkpoint_eval, random_eval, build_eval_function


FICTITIOUS_CONFIG = {
    # The RLLib Trainable to run for each agent
    "alg": "PPO",
    # Whether the game should be treated as symmetric (only train a single policy in this case)
    "symmetric": False,
    # The maximum number of cycles over all agents for self play
    "max_cycles": 5,
    # Stopping conditions for self-play rounds
    "self_play_round_stop": {
        "training_iteration": 10
    },
    # Stopping conditions for pretraining
    "self_play_pretrain_stop": {
        "training_iteration": 0
    },
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


def build_wrapper_cls(env_creator):
    
    class RandomWrapper(MultiAgentEnv):
        """
        This class wraps an existing multi-agent env, allowing us to randomize
        the policy mapping for each episode.  Needed for fictitious self-play.
        """

        def __init__(self, config):
            self._env = env_creator(config)
            self._forward_mapping = {pid:pid for pid in self._env.action_space_dict.keys()}
            self._forward_mapping["__all__"] = "__all__"  # Add mapping for "__all__" for "done" values
            self._reverse_mapping = self._forward_mapping.copy()
            self._mixture = None
            
        def _forward(self, env_dict):
            return {self._forward_mapping[key]: value for key, value in env_dict.items()}

        def _reverse(self, action_dict):
            return {self._reverse_mapping[key]: value for key, value in action_dict.items()}

        def set_mixture(self, mixture):
            self._mixture = mixture.copy()

        def reset(self):
            for agent_id, policy_ids in self._mixture:
                self._forward_mapping[agent_id] = np.random.choice(policy_ids)

            for key, value in self._forward_mapping.items():
                    self._reverse_mapping[value] = key
            
            return self._forward(self._env.reset())

        def step(self, action_dict):
            obs, rew, done, info = self._env.step(self._reverse(action_dict))
            return self._forward(obs), self._forward(rew), self._forward(done), self._forward(info)
    
    return UpdateWrapper


class FictitiousPlay(Trainable):

    def _setup(self, config):
        config = merge_dicts(FICTITIOUS_CONFIG, config)
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

        # Check if we are doing symmetric or asymmetric training
        self._symmetric = config.pop("symmetric")
        self._round_stop = config.pop("self_play_round_stop")
        self._pretrain_stop = config.pop("self_play_pretrain_stop")
        self._max_cycles = config.pop("max_cycles")

        # Build multi-agent configs for training and evaluation
        if self._symmetric:
            raise NotImplementedError("Symmetric training is not implemented yet")
        else:
            policies = dict()

            # Define the full set of policies
            for pid in env.observation_space_dict.keys():
                for cycle in range(self._max_cycles)
                policies[f"learned_policy_{pid}"] = (
                        None,
                        env.observation_space_dict[pid],
                        env.action_space_dict[pid],
                        {}
                    )

            # Define the set of policy mixtures
            self._policy_mixtures
        
            policies_to_train = list(policies.keys()) 
            # Define the initial training set - we pretrain all policies at once
            self._policy_cycle = cycle(policies_to_train)

        # Get config for population based evaluation
        checkpoints = config.pop("population")
        policies.update(make_checkpoint_config(checkpoints))

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

        config["evaluation_config"] = {
            "multiagent": {
                "policies": policies,
                "policies_to_train": [],
                "policy_mapping_fn": lambda pid: pid,
            }
        }

        # Define evaluation config
        config["evaluation_num_workers"] = config.pop("multiagent_eval_num_workers")
        config["evaluation_interval"] = config.pop("multiagent_eval_interval")

        # Build custom evaluation function
        base_mapping = {idx: policy for idx, policy in enumerate(policies_to_train)}
        eval_episodes = config.pop("multiagent_eval_episodes")
        config["custom_eval_function"] = build_eval_function(checkpoints, base_mapping, eval_episodes, random_eval)

        # Build the base Trainable
        trainer_cls = get_trainable_cls(config.pop("alg"))
        self._trainer = trainer_cls(config, 
                            env=build_wrapper_cls(env_creator), 
                            logger_creator=lambda config: NoopLogger(config, self._logdir))

        # Copy checkpoint weights to evaluation policies
        checkpoint_weights = dict()
        for checkpoint in checkpoints:
            checkpoint_weights[checkpoint.policy_id] = ray.get(checkpoint.weights)

        self._trainer.workers.local_worker().set_weights(checkpoint_weights)

        # Initialize self-play state parameters
        self._is_pretraining = True
        self._self_play_round = 0
        self._round_stats = {
            TRAINING_ITERATION: 0,
            TIMESTEPS_TOTAL: 0, 
            TIME_TOTAL_S: 0, 
            EPISODES_TOTAL: 0,
        }

    def _start_round(self):
        """ Begins a new round of self play. """
        
        # Increment the round counter
        self._self_play_round += 1

        # Reset round statistics
        self._round_stats = {
            TRAINING_ITERATION: 0,
            TIMESTEPS_TOTAL: 0, 
            TIME_TOTAL_S: 0, 
            EPISODES_TOTAL: 0,
        }

        if self._symmetric:
            # TODO: If symmetric, move weights from learned policy to fixed policy, 
            raise NotImplementedError("Symmetric training is not implemented yet")
        else:
            # Otherwise, change the policy which is currently training
            self._trainer.workers.local_worker().policies_to_train = [next(self._policy_cycle)]

    def _is_round_finished(self, stop_conditions):
        for key, value in stop_conditions.items():
            if self._round_stats[key] >= value:
                return True

        return False

    def _train(self):
        
        # Check if the current round is finished
        if self._is_pretraining:
            if self._is_round_finished(self._pretrain_stop):
                self._is_pretraining = False
                self._start_round()
        elif self._is_round_finished(self._round_stop):
            self._start_round()

        results = self._trainer.train()

        # Update round statistics
        self._round_stats[TIMESTEPS_TOTAL] += results[TIMESTEPS_THIS_ITER]
        self._round_stats[EPISODES_TOTAL] += results[EPISODES_THIS_ITER]
        self._round_stats[TIME_TOTAL_S] += results[TIME_THIS_ITER_S]
        self._round_stats[TRAINING_ITERATION] += 1

        results["self_play_rounds"] = self._self_play_round
        results["round_stats"] = self._round_stats.copy()
    
        return results

    def _save(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "self_play_state.pkl"), "wb") as f:
            cloudpickle.dump({
                    "round_stats": self._round_stats,
                    "self_play_round": self._self_play_round,
                    "is_pretraining": self._is_pretraining,
                    "policy_cycle": self._policy_cycle,
                }, f)
        return self._trainer.save(checkpoint_dir)

    def _restore(self, checkpoint):
        with open(os.path.join(checkpoint, "../self_play_state.pkl"), "rb") as f:
            state = cloudpickle.load(f)

        self._round_stats = state["round_stats"]
        self._self_play_round = state["self_play_round"]
        self._is_pretraining = state["is_pretraining"]
        self._policy_cycle = state["policy_cycle"]
        
        self._trainer.restore(checkpoint)

    def _stop(self):
        self._trainer.stop()
