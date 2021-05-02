#!/usr/bin/env python3

'''
Script for running self-play with evaluation against a pre-trained populations of agents
'''

import argparse
import numpy as np
import yaml

import ray
from ray.tune import run_experiments

import algorithms
from algorithms.self_play import load_checkpoints
import environments


DEFAULT_EXPERIMENT = {
    "DEEP_NASH_RND_markov_soccer": {
        "run": "SIMULTANEOUS_PLAY",
        "stop": {
            "timesteps_total": 1000000,
        },
        "checkpoint_freq": 100,
        "checkpoint_at_end": True,
        "num_samples": 3,
        "config": {
            "alg": "DEEP_NASH_V1",
            "population": [
                {
                    "path": "baselines/openspiel/markov_soccer",
                    "alg": "PPO",
                    "mapping": [(0, "policy_0")],
                },
            ],
            "random_eval": True,
            "evaluation_interval": 1,
            "env": "openspiel",
            "env_config": {
                "game": "markov_soccer",
            },
            "horizon": 200,
            "gamma": 0.99,
            "lr": 0.0001,
            "rollout_fragment_length": 400,
            "evaluation_num_workers": 2,  # Must be at least one for Deep Nash
            "batch_mode": "complete_episodes",

            # === Curiosity ===
            "model": {
                "custom_options": {
                    "scale": 0.01,
                    "weight": 0.5,
                    "burn_in": 20000,
                    "delay": 10000,
                    "gamma": 0.95,
                    "curiosity_module": "RND",
                    "curiosity_config": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [32, 32],
                        "fcnet_outputs": 32,
                    },
                },
            },
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser("General purpose multi-agent training script enabling evaluation against a pre-trained population.")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from these files.")
    parser.add_argument("--local-dir", type=str, default="../ray_results_population",
                        help="path to save training results")
    parser.add_argument("--num-cpus", type=int, default=7,
                        help="the maximum number of cpus ray is allowed to us, useful for running locally")
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
        experiments = dict()

        for config_file in args.config_file:
            with open(config_file) as f:
                experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        experiments = DEFAULT_EXPERIMENT

    # Initialize ray - has to be done before we load the checkpoints
    ray.init(num_cpus=args.num_cpus)

    # Update experiment configs
    for experiment in experiments.values():
        exp_config = experiment["config"]

        # Load checkpoints
        if "population" in exp_config:
            exp_config["population"] = load_checkpoints(exp_config["population"])

        # Set num workers
        exp_config["num_workers"] = args.num_workers

        # Add intrinsic reward logging
        exp_config["callbacks"] = {
            "on_postprocess_traj": log_intrinsic
        }  

        # Set local directory for checkpoints
        experiment["local_dir"] = str(args.local_dir)

        # Modify config to reduce TensorFlow thrashing
        exp_config["tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

        exp_config["local_tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

    # Run experiments
    run_experiments(experiments, verbose=1, reuse_actors=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
