#!/usr/bin/env python3

'''
Utility script for selecting the best configuration from an RLLib hyperparamter tuning run.
'''

import argparse
import os

from ray.tune import Analysis, ExperimentAnalysis


def parse_args():
    parser = argparse.ArgumentParser("Identifies the best hyperparameters settings from a set of RLLib experiments")
    
    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("--key", type=str, default="episode_reward_mean", 
        help="key of the RLLib metric to sort on")
    parser.add_argument("--mode", type=str, default="max",
        help="whether to maximize or minimize the given key ['max','min']")
    
    return parser.parse_args()


def get_best_configs(trials, key, mode='max'):
    means = []

    for name, data in trials.items():
        means.append((name, data[key].mean(),))

    if 'max' == mode:
        target = max([mean for name, mean in means])
    elif 'min' == mode:
        target = min([mean for name, mean in means])
    else:
        return ValueError(f"Mode '{mode}' is undefined")

    return [name for name, mean in means if mean == target]


def main(args):
    print(f"Loading '{args.path}'")

    analysis = Analysis(args.path)
    dataframes = analysis.trial_dataframes

    best_configs = get_best_configs(dataframes, args.key, mode=args.mode)

    print(f"\n\n\n===== {args.mode} {args.key} =====\n")

    for config in best_configs:
        print(config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
