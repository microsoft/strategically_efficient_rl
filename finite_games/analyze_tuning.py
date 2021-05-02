#!/usr/bin/env python3

'''
Script to select the best configuration from a hyperparameter sweep.
'''

import argparse
import json
import numpy as np
import os
import os.path
import pandas
import yaml

from grid_search import grid_search


class Configuration:

    def __init__(self, params):
        self.params = params
        self.runs = []


def parse_args():
    parser = argparse.ArgumentParser("Identifies the best hyperparameters settings from a tuning sweep")
    
    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-l", "--loss", type=str, default="nash_conv", 
        help="key of the metric to minimize (or maximize)")
    parser.add_argument("-a", "--accumulate", type=str, default="mean", 
        help="method for condensing time series into a scalar ['mean','max','min']")
    parser.add_argument("-m", "--mode", type=str, default="min",
        help="whether to maximize or minimize the given key ['max','min']")
    
    return parser.parse_args()


def load_variations(path):
    with open(os.path.join(path, "config.json"), 'r') as config_file:
        config = json.load(config_file)
    
    return grid_search(config)


def load_runs(path, loss, accumulate):
    print(f"loading: {path}")
    runs = []

    if os.path.isdir(path):
        for obj in os.listdir(path):
            results_path = os.path.join(path, obj)

            if os.path.isdir(results_path):
                results_file = os.path.join(results_path, "results.csv")

                if os.path.isfile(results_file):
                    results = pandas.read_csv(results_file)

                    # Filter out empy data series
                    if results.shape[0] > 0:
                        result = results[loss]

                        if results.shape[0] > 0 and not np.any(np.isnan(result)):
                            if "max" == accumulate:
                                value = np.max(result)
                            elif "max" == accumulate:
                                value = np.min(result)
                            else:
                                value = np.mean(result)

                            runs.append(value)                        

    return runs


def main(args):
    print(f"Path: {args.path}")
    print("Loading runs...")

    # Load variations
    with open(os.path.join(args.path, "config.json"), 'r') as config_file:
        experiments = json.load(config_file)

    variations = {}

    for name, config in experiments.items():
        variations.update(grid_search(name, config))

    # Collect all runs for each config
    configs = dict()

    for name, config in variations.items():
        runs = load_runs(os.path.join(args.path, name), args.loss, args.accumulate)
        config_str = json.dumps({
            "alg": config["alg"],
            "alg_config": config["alg_config"]
        }, sort_keys=True)

        if config_str not in configs:
            configs[config_str] = Configuration(config["alg_config"])

        configs[config_str].runs.extend(runs)

    # Identify best configuration
    if "min" == args.mode:
        best_mean = np.Infinity
    else:
        best_mean = -np.Infinity
    
    best_configs = []

    for config in configs.values():
        if len(config.runs) > 0:
            mean = np.mean(config.runs)

            print("\n------------")
            print(f"Mean: {mean}")
            print("Config:")
            print(yaml.dump(config.params, default_flow_style=False))

            if mean == best_mean:
                best_configs.append(config.params)
            elif "min" == args.mode:
                if mean < best_mean:
                    best_mean = mean
                    best_configs = [config.params]
            else:
                if mean > best_mean:
                    best_mean = mean
                    best_configs = [config.params]
    
    # Return best config
    print(f"\nBest Value: {best_mean}")
    print("Best Configs:")

    for config in best_configs:
        print("\n----------\n")
        print(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    args = parse_args()
    main(args)
