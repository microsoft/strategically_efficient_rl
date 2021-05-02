#!/usr/bin/env python3

'''
Uses a specified algorithm to compute the equilibrium of an initially unknown zero-sum game in normal form.
'''

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import os.path
import pandas

from normal_form.games import build_game
from normal_form.algorithms import build_algorithm


def parse_args():
    parser = argparse.ArgumentParser("Computes an equilibrium of equilibrium of an unknown, zero-sum game in normal form")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/normal_form",
                        help="directory in which we should save results")
    parser.add_argument("--steps", default=10000, type=int,
                        help="The number of samples to train for")
    parser.add_argument("--eval-interval", default=100, type=int,
                        help="Interval between strategy evaluations")
    parser.add_argument("--num-games", default=1, type=int,
                        help="Number of randomly generated games to train on")
    parser.add_argument("--num-runs", default=1, type=int,
                        help="Number of runs per game")

    return parser.parse_args()


def load_configs(args):
    experiments = dict()

    if args.config_file:
        for config_file in args.config_file:
            with open(config_file) as f:
                experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        experiments = {
            "nash_v_hybrid": {
                "rows": 10,
                "columns": 10,
                "game": "hybrid_roshambo",
                "game_config": { "cyclic_actions": 3, },
                "alg": "nash_v",
                "alg_config": {},
                "steps": 200000,
                "eval_interval": 1000,
                "num_games": 10,
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


def entropy(strategy):
    entropy =0.0

    for probability in strategy:
        if probability > 0.0:
            log = np.log(probability)

            if np.isfinite(log):
                entropy -= probability * log

    return entropy


def main(args):
    experiments = load_configs(args)

    for name, experiment in experiments.items():

        # Make results directory
        path = make_unique_dir(args.output_path, name)

        # Save configuration
        with open(os.path.join(path, "config.json"), "w") as config_file:
            json.dump({name: experiment}, config_file)

        for game_idx in range(experiment["num_games"]):
            game = build_game(experiment["game"], 
                              experiment["rows"], 
                              experiment["columns"],
                              experiment.get("game_config", {}))

            for run_idx in range(experiment["num_runs"]):
                learner = build_algorithm(experiment["alg"], 
                                          game.N,
                                          game.M, 
                                          experiment["steps"], 
                                          experiment.get("alg_config", {}))

                results = defaultdict(list)
                
                for sample in range(experiment["steps"]):
                    learner.sample(game)

                    if sample != 0 and sample % experiment["eval_interval"] == 0:
                        print(f"{name} - game {game_idx}, run {run_idx} - sample {sample}")

                        row_strategy, column_strategy = learner.strategies()
                        row_entropy = entropy(row_strategy)
                        column_entropy = entropy(column_strategy)

                        if np.isnan(row_entropy) or np.isnan(column_entropy):
                            raise ValueError("row or column entropy is NaN")

                        row_value, column_value, nash_conv = game.nash_conv(row_strategy, column_strategy)

                        results["samples"].append(sample)
                        results["row_value"].append(row_value)
                        results["column_value"].append(column_value)
                        results["nash_conv"].append(nash_conv)
                        results["row_entropy"].append(row_entropy)
                        results["column_entropy"].append(column_entropy)
                        results["total_entropy"].append(row_entropy + column_entropy)

                # Build and save data frame
                results_file = f"{name}_{game_idx}_{run_idx}.csv"
                results_file = os.path.join(path, results_file)

                dataframe = pandas.DataFrame(results)
                dataframe.to_csv(results_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
