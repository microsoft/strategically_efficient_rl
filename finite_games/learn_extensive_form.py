#!/usr/bin/env python3

'''
Training script for finite, constant-sum extensive-form games.
'''

import argparse
from collections import defaultdict
import json
from multiprocessing import Pool
import numpy as np
import os
import os.path
import pandas
from tensorboardX import SummaryWriter
import traceback
import yaml

from extensive_form.games import build_game
from extensive_form.algorithms import build_algorithm
from extensive_form.nash_conv import nash_conv
from grid_search import grid_search


def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    parser.add_argument("-n", "--num-cpus", type=int, default=4,
                        help="the number of parallel worker processes to launch")

    return parser.parse_args()


def load_configs(args):
    if args.config_file:
        experiments = dict()

        for config_file in args.config_file:
            with open(config_file) as f:
                experiments.update(yaml.load(f, Loader=yaml.FullLoader))

    else:
        experiments = {
            "optimistic_nash_q_decoy_deep_sea": {
                "game": "decoy_deep_sea",
                "game_config": {
                    "decoy_games": 20,
                    "decoy_size": 20,
                    "decoy_payoff": 1.,
                    "target_size": 20,
                    "target_payoff": 1.,
                    "target_penalty": 0.0,
                    "adversary_payoff": 1.0
                },
                "alg": "strategic_ulcb",
                "alg_config": {
                    "iteration_episodes": 10,
                    "beta": 1.0,
                },
                "iterations": 200,
                "eval_interval": 10,
                "num_games": 1,
                "num_runs": 1,
            }
        }

    return experiments


def make_unique_dir(path, tag):
    sub_path = os.path.join(path, tag)
    idx = 0

    while os.path.exists(sub_path):
        idx += 1
        sub_path = os.path.join(path, tag + "_" + str(idx))
    
    os.makedirs(sub_path)
    return sub_path


def run_trail(path, experiment, game, game_idx, run_idx, seed):
    path = os.path.join(path, f"game_{game_idx}_run_{run_idx}")
    os.makedirs(path)

    # Initialize tensorboard logging
    writer = SummaryWriter(path, flush_secs=30)

    # Reseed numpy for parallel execution
    np.random.seed(seed)

    # Build environment and learning algorithm
    learner = build_algorithm(experiment["alg"], 
                              game.build_env(),
                              experiment.get("alg_config", {}))

    results = defaultdict(list)
    statistics = defaultdict(list)
    total_samples = 0
    total_episodes = 0
    
    for iteration in range(experiment["iterations"]):
        samples, episodes, stats = learner.train()
        total_samples += samples
        total_episodes += episodes

        # Accumulate statistics between evaluation rounds
        for key, value in stats.items():
            statistics[key].append(value)

        if iteration != 0 and iteration % experiment["eval_interval"] == 0:

            # Compute NashConv end eval statistic
            row_value, column_value, nash = nash_conv(game, learner)

            results["iterations"].append(iteration)
            results["samples"].append(total_samples)
            results["episodes"].append(total_episodes)
            results["row_value"].append(row_value)
            results["column_value"].append(column_value)
            results["nash_conv"].append(nash)

            # Compute mean episode statistics
            for key, value in statistics.items():
                results["stats/" + key].append(np.mean(value))

            statistics = defaultdict(list)

            # Log to tensorboard
            for key, values in results.items():
                if not key.startswith("stats/"):
                    key = "loss/" + key

                writer.add_scalar(key, values[-1], global_step=results["samples"][-1])

    # Build and save data frame
    results_file = os.path.join(path, "results.csv")

    dataframe = pandas.DataFrame(results)
    dataframe.to_csv(results_file)

    # Close tensorboard writer
    writer.close()


def run_experiment(output_path, name, experiment, pool):

    # Make results directory
    path = make_unique_dir(output_path, name)

    # Save configuration
    with open(os.path.join(path, "config.json"), 'w') as config_file:
        json.dump({name: experiment}, config_file)

    # Launch trials
    trials = []
    seed = 0

    for game_idx in range(experiment["num_games"]):
        
        # Reseed to ensure games are identical across experiments
        np.random.seed(game_idx)

        # Build new (randomly generated) game
        game = build_game(experiment["game"], experiment.get("game_config", {}))

        for run_idx in range(experiment["num_runs"]):
            print(f"{name} - game: {game_idx}, run: {run_idx}")
            trials.append(pool.apply_async(run_trail, (path, experiment, game, game_idx, run_idx, seed), error_callback=print_error))
            seed += 1
    
    return trials


def main(args):
    experiments = load_configs(args)
    pool = Pool(args.num_cpus)

    trials = []

    for name, experiment in experiments.items():
        variations = grid_search(name, experiment)

        if variations is not None:
            output_path = make_unique_dir(args.output_path, name)

            with open(os.path.join(output_path, "config.json"), 'w') as config_file:
                json.dump({name: experiment}, config_file)

            for var_name, var_experiment in variations.items():
                trials += run_experiment(output_path, var_name, var_experiment, pool)
        else:
            trials += run_experiment(args.output_path, name, experiment, pool)

    for trial in trials:
        trial.wait()


if __name__ == '__main__':
    args = parse_args()
    main(args)
