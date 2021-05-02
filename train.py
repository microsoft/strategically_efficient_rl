#!/usr/bin/env python3

'''
Generic RLLib training script that logs intrisic reward if present.
'''

import argparse
import yaml

import numpy as np
import ray
from ray.tune import run_experiments

import algorithms
import environments


def parse_args():
    parser = argparse.ArgumentParser("Generic training script for any registered single agent environment, logs intrinsic reward stats.")

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
            "PPO_Pong": {
                "run": "PPO",
                "env": "Pong-v0",
                "stop": {
                    "timesteps_total": 100000,
                },
                "checkpoint_freq": 10,
                "local_dir": "../ray_results",
                "num_samples": 3,
                "config": {
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "entropy_coeff": 0.001,
                    "clip_param": 0.1,
                    "lr": 0.001,
                    "num_sgd_iter": 4,
                },
            },
        }

    # Add intrinsic reward logging
    for experiment in EXPERIMENTS.values():

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

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    run_experiments(EXPERIMENTS, verbose=0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
