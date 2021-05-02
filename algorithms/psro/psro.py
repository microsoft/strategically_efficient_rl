"""
An RLLib implementation of PSRO (Lanctot et. al. 2017) for two-player games.  Takes the meta-strategy as a 
configuration parameter, supporting self-play, fictitious-play, and Nash response.

NOTE: Does not currently support the Nash-response meta-strategy.

Also supports centralized critics and curiosity modules, and population-based evaluation.
"""

import argparse
import copy
from collections import namedtuple
import numpy as np
import yaml

import ray
from ray.tune.registry import ENV_CREATOR, register_env, _global_registry
from ray.tune import Trainable
from ray.tune.result import TRAINING_ITERATION, TIMESTEPS_TOTAL, TIME_TOTAL_S, EPISODES_TOTAL
from ray.tune import run_experiments
from ray.tune.registry import register_env, register_trainable
from ray.rllib.agents.registry import get_agent_class


DEFAULT_CONFIG = {
    "symmetric": False,
    "alg": "DQN",
    "alg_config": {
        "gamma": 0.95,
        "horizon": 100, # Controls episode length (independent of step limits in the environment)
    },
    "red_team_ids": ["0", "1"],  # TODO: Allow teams to be specified using the RLLib grouping mechanism
    "blue_team_ids": ["2"],
    # Should support any RLLib stopping criterion, not sure if we can do adaptive stopping (based on convergence)
    "round_stop": {
        "training_iteration": 100
    },
    # Stopping conditions for pretraining rounds
    "pretrain_round_stop": {
        "training_iteration": 100
    },
    "schedule_max_timesteps": 100000,
    "meta_strategy": "latest"
}


# This is going to hurt, but we need a custom trainable that implements self play, wrapping an underlying trainable
class PSRO(Trainable):

    def _setup(self, config):

        # Move environment configuration into the algorithm config - The individual Trainers need to instantiate their environment
        env_name = config["env"]
        env_config = config.get("env_config", {})

        self._algorithm_config = config.get("alg_config", {})
        self._algorithm_config["env"] = env_name
        self._algorithm_config["env_config"] = env_config

        # Build an instance of the environment so we can extract the observation and action spaces
        if _global_registry.contains(ENV_CREATOR, env_name):
            env_creator = _global_registry.get(ENV_CREATOR, env_name)
        else:
            import gym
            env_creator = lambda env_config: gym.make(env_name)

        env = env_creator(env_config)

        self._observation_spaces = env.observation_space_dict
        self._action_spaces = env.action_space_dict

        # Get the trainer class
        self._trainer_cls = get_agent_class(config["alg"])

        # Initialize data structures to hold policy checkpoints
        self._red_team = []
        self._blue_team = []

        self._current_training_team = None

        # Get the meta-strategy - How do we end up using the meta stategy? - Here the meta-strategy is just a keyword
        self._meta_strategy = config.get("meta_strategy", "random")
       
        # Initialzie round counter and round termination conditions
        self._current_results = None
        self._self_play_round = 0

        self._training_iteration = 0
        self._episodes_total = 0
        self._timesteps_total = 0
        self._time_total_s = 0

        self._round_stop = self.config.get("round_stop", {"training_iteration": 10})

        self._pretrain_round_stop = config.get("pretrain_round_stop", {"training_iteration": 10})
        self._is_pretraining = True

        # Define policy mapping function
        self._policy_mapping_fn = lambda idx: str(idx)

        # Intialize the pretrainer - we always do pretraining, we may just do zero iterations of it
        self._trainer = self._trainer_cls(self._build_trainer_config())

    def _build_trainer_config(self, fixed_policies=dict()):  # This doesn't need to be re-done for each epoch, only policies_to_train needs to change
        """ Constructs a trainer config dictionary where a subset of agent policies are fixed """
    
        # Copy config
        config = copy.deepcopy(self._algorithm_config)

        # Build policy dictionary - we will need a special case for Symmetric games
        policies = dict()
        policies_to_train = []

        for policy_id in self._observation_spaces.keys():
            policy_id_str = self._policy_mapping_fn(policy_id)  # Base ids are ints, need to be strings

            if policy_id_str not in fixed_policies:
                policies_to_train.append(policy_id_str)

            policies[policy_id_str] = (None, self._observation_spaces[policy_id], self._action_spaces[policy_id], {})

        # Configure policy map - we don't need to know which environment we are looking at to do this
        config["multiagent"] = {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": self._policy_mapping_fn  # Seems that for the particle env ids are just integers  - converted to strings
        }

        return config

    def _start_round(self, training_team, fixed_team):
        """ begins a new round of self play. """

        # Stop previous experiment
        self._trainer.stop()

        # Update running totals based on previous results
        self._self_play_round += 1

        self._training_iteration += self._current_results[TRAINING_ITERATION]
        self._episodes_total += self._current_results[EPISODES_TOTAL]
        self._timesteps_total += self._current_results[TIMESTEPS_TOTAL]
        self._time_total_s += self._current_results[TIME_TOTAL_S]

        # Sample checkpoint and extract weights
        if self._meta_strategy == "latest":
            fixed_checkpoint = fixed_team[-1]
        else:
            fixed_checkpoint = np.random.choice(fixed_team)

        # Get training policies
        training_checkpoint = training_team[-1]

        # Construct new trainer - initialize weights for policies being trained - for fictitious play, need to store all previous policies in the trainer
        # At each round we will need to build a new Trainer config, with more policies than agents
        self._trainer = self._trainer_cls(self._build_trainer_config(fixed_checkpoint))
        self._trainer.set_weights(fixed_checkpoint)
        self._trainer.set_weights(training_checkpoint)

        # Wipe out previous results
        self._current_results = None

    def _is_round_finished(self, stop_conditions):
        if self._current_results is not None:
            for key, value in stop_conditions.items():
                if self._current_results[key] >= value:
                    return True

        return False

    def _train(self):

        # Check if we are still doing pretraining - if so, and pretraining is finished
        if self._is_pretraining and self._is_round_finished(self._pretrain_round_stop):
                self._is_pretraining = False

                # Extract team policies from the pretrainer - should we bother with teams?
                self._red_team.append(self._trainer.get_weights(self.config["red_team_ids"]))
                self._blue_team.append(self._trainer.get_weights(self.config["blue_team_ids"]))

                # Start new round - red team first
                self._start_round(self._red_team, self._blue_team)
        elif self._is_round_finished(self._round_stop):
            if self._current_training_team == self._red_team:
                self._red_team.append(self._trainer.get_weights(self.config["red_team_ids"]))
                self._start_round(self._blue_team, self._red_team)
            else:
                self._red_team.append(self._trainer.get_weights(self.config["blue_team_ids"]))
                self._start_round(self._red_team, self._blue_team)

        # Do training iteration - save latest results to check for termination condition
        self._current_results = self._trainer.train()

        # Override cummulative totals in results dict to ensure that external termination conditions are triggered
        results = copy.deepcopy(self._current_results)
        results["self_play_round"] = self._self_play_round

        # These may get overriden already
        results[TRAINING_ITERATION] += self._training_iteration
        results[EPISODES_TOTAL] += self._episodes_total
        results[TIMESTEPS_TOTAL] += self._timesteps_total
        results[TIME_TOTAL_S] + self._time_total_s
    
        return results

    def _save(self, tmp_checkpoint_dir):
        return self._trainer._save(tmp_checkpoint_dir)

    def _stop(self):
        self._trainer.stop()
