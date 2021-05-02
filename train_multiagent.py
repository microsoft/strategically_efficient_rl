#!/usr/bin/env python3

'''
Trains independent learners in a multi-agent environment.  Logs intrinsic reward if present.
'''

import argparse
import yaml

import numpy as np
import ray
from ray.tune import run_experiments
from ray.tune.registry import ENV_CREATOR, _global_registry

import algorithms
import environments


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
    parser.add_argument("--num-workers", type=int, default=3,
                        help="the number of parallel workers per experiment")
    parser.add_argument("--nash-conv", action="store_true",
                        help='compute and record NashConv losses if environment supports this')

    
    return parser.parse_args()


def nash_conv(trainer, eval_workers):
    local_worker = eval_workers.local_worker()

    env = local_worker.env_creator(local_worker.policy_config["env_config"])
    mapping_fn = local_worker.policy_config["multiagent"]["policy_mapping_fn"]

    if hasattr(env, "nash_conv"):
        policy_dict = {pid: local_worker.get_policy(mapping_fn(pid)) for pid in env.action_space_dict.keys()}
        return env.nash_conv(policy_dict)


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
    EXPERIMENTS = dict()

    if args.config_file:
        for config_file in args.config_file:
            with open(config_file) as f:
                EXPERIMENTS.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        EXPERIMENTS = {
            "RND_deep_sea": {
                "run": "PPO_CURIOSITY",
                "stop": {
                    "timesteps_total": 500000,
                },
                "checkpoint_at_end": True,  # Make sure we have a final checkpoint for evaluation
                "config": {
                    "env": "roshambo",
                    "env_config": {},
                    "model": {
                        "custom_options": {
                            "weight": 0.5,
                            "decay": 0.02,
                            "burn_in": 10000,
                            "delay": 5000,
                            "curiosity_module": "RND",
                            "curiosity_config": {
                                "fcnet_activation": "relu",
                                "fcnet_hiddens": [256, 256],
                                "fcnet_outputs": 32,
                                "agent_action": True,
                                "joint_action": False,
                            },
                        },
                    },
                    "horizon": 1000,
                    "intrinsic_gamma": 0.95,
                    "intrinsic_lambda": 0.95,
                    "num_agents": 1,  # Controls whether we are using multi-agent or single-agent curiosity
                    "lambda": 0.95,
                    "gamma": 0.95,
                    "entropy_coeff": 0.001,
                    "clip_param": 0.1,
                    "lr": 0.001,
                    "num_sgd_iter": 8,
                    "num_workers": 4,
                    "train_batch_size": 4000,
                    "rollout_fragment_length": 500,
                },
            },
        }

    # Add intrinsic reward logging and multiagent configuration
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

        # One policy per agent for multiple individual learners  # NOTE: Will this work for single-agent environments?
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

        # Define custom NashConv evaluation function if needed
        if args.nash_conv:
            exp_config["custom_eval_function"] = nash_conv
            exp_config["evaluation_config"] = exp_config.get("evaluation_config", {
                "in_evaluation": True,
            })

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

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    run_experiments(EXPERIMENTS, verbose=1)

if __name__ == '__main__':
    args = parse_args()
    main(args)
