#!/usr/bin/env python3

'''
Alternative implementation of self-play using Ray Tune custom schedulers
'''

import argparse
from collections import defaultdict
import yaml

import ray
from ray.tune import run_experiments
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL, EPISODES_TOTAL, TRAINING_ITERATION
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial, Checkpoint

import algorithms
import environments


class SelfPlayTrial:

    def __init__(self, trial, trial_runner, config):
        self._trial = trial
        self._trial_runner = trial_runner
        self._config = config

        # Initialize iteration counter
        self._self_play_round = 0

        # Initialize previous result
        self._last_update = defaultdict(lambda: 0)

        # Get multiagent config
        multiagent = trial.config["multiagent"]

        if "policies_to_train" in multiagent:
            self._policy_ids = multiagent["policies_to_train"]
        else:
            self._policy_ids = list(multiagent["policies"].keys())

        if "initial_policy" in config:
            self._current_policy = self._policy_ids.index(config["initial_policy"])
        else:
            self._current_policy = 0

        self._is_burn_in = config.get("burn_in", False)

        # Initialize training policies
        if not self._is_burn_in:
            multiagent["policies_to_train"] = [self._policy_ids[self._current_policy]]

    def _update(self, result):
        self._self_play_round += 1

        # Update training policy
        if self._is_burn_in:
            self._is_burn_in = False
        else:
            self._current_policy = (self._current_policy + 1) % len(self._policy_ids)

        # Update last result
        for key in self._config["round"].keys():
            self._last_update[key] = result[key]

        # Save current state
        checkpoint = self._trial_runner.trial_executor.save(self._trial, Checkpoint.MEMORY, result=result)

        # Update config
        self._trial.config["multiagent"]["policies_to_train"] = [self._policy_ids[self._current_policy]]

        # Reset trial  
        self._trial_runner.trial_executor.reset_trial(self._trial, self._trial.config, self._trial.experiment_tag)        

        # Restore
        self._trial_runner.trial_executor.restore(self._trial, checkpoint)

    def on_result(self, result):
        for key, value in self._config["round"].items():
            if result[key] - self._last_update[key] >= value:
                self._update(result)
                break
        
        result["self_play_round"] = self._self_play_round

        return TrialScheduler.CONTINUE


class SelfPlayScheduler(FIFOScheduler):

    def __init__(self):
        self._trials = {}

    def on_trial_add(self, trial_runner, trial):
        if "self_play" in trial.config:
            self._trials[trial] = SelfPlayTrial(trial, trial_runner, trial.config.pop("self_play"))

    def on_trial_result(self, trial_runner, trial, result):
        return self._trials[trial].on_result(result)

    def debug_string(self):
        return "Using custom scheduler to implement self play"


def parse_args():
    parser = argparse.ArgumentParser("Generic training script for any registered multi-agent environment, logs intrinsic reward stats")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from these files.")
    parser.add_argument("--local-dir", type=str, default="../ray_results",
                        help="path to save training results")
    parser.add_argument("--num-cpus", type=int, default=7,
                        help="the maximum number of CPUs ray is allowed to us, useful for running locally")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="the maximum number of GPUs ray is allowed to use")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="the number of parallel workers per experiment")
    
    return parser.parse_args()


def log_intrinsic(info):
    episode = info["episode"]
    batch = info["post_batch"]

    if "intrinsic_reward" in batch:
        reward = np.sum(batch["intrinsic_reward"])

        if "intrinsic_reward" not in episode.custom_metrics:
            episode.custom_metrics["intrinsic_reward"] = reward
        else:
            episode.custom_metrics["intrinsic_reward"] += reward


def main(args):
    if args.config_file:
        EXPERIMENTS = dict()

        for config_file in args.config_file:
            with open(config_file) as f:
                EXPERIMENTS.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        EXPERIMENTS = {
            "PPO_KeepAway": {
                "run": "PPO",
                "stop": {
                    "timesteps_total": 20000,
                },
                "checkpoint_freq": 10,
                "num_samples": 1,
                "config": {
                    "self_play": {
                        "burn_in": True,
                        "round": {
                            "timesteps_total": 2000,
                        },
                    },
                    "horizon": 200,
                    "env": "roshambo",
                    "env_config": {},
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "entropy_coeff": 0.001,
                    "clip_param": 0.1,
                    "lr": 0.001,
                    "num_sgd_iter": 4,
                    "train_batch_size": 400,
                    "rollout_fragment_length": 200,
                },
            },
        }

    # Add multiagent configurations
    for experiment in EXPERIMENTS.values():
        exp_config = experiment["config"]

        # Create temporary env instance to query observation space, action space and number of agents
        env_name = exp_config["env"]

        if _global_registry.contains(ENV_CREATOR, env_name):
            env_creator = _global_registry.get(ENV_CREATOR, env_name)
        else:
            import gym
            env_creator = lambda env_config: gym.make(env_name)

        env = env_creator(exp_config.get("env_config", {}))

        # One policy per agent for multiple individual learners
        policies = dict()
        for pid in env.observation_space_dict.keys():
            policies[f"policy_{pid}"] = (
                None,
                env.observation_space_dict[pid],
                env.action_space_dict[pid],
                {}
            )
  
        exp_config["multiagent"] = {"policies": policies,
                                    "policy_mapping_fn": lambda pid: f"policy_{pid}"}

        # Set local directory for checkpoints
        experiment["local_dir"] = str(args.local_dir)

        # Set num workers
        experiment["config"]["num_workers"] = args.num_workers

        # Add intrinsic reward logging
        experiment["config"]["callbacks"] = {
            "on_postprocess_traj": log_intrinsic
        }

        # Modify config to reduce TensorFlow thrashing
        experiment["config"]["tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

        experiment["config"]["local_tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

    # Build scheduler
    scheduler = SelfPlayScheduler()

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    run_experiments(EXPERIMENTS, verbose=1, scheduler=scheduler)

if __name__ == '__main__':
    args = parse_args()
    main(args)