#!/usr/bin/env python3

'''
Utility script for generating plots from data stored in RLLib-generated CSV files
'''

import argparse
from collections import namedtuple
import os
import pandas  # NOTE: Pandas may not be an option, as it seems to throw a security error - even if this script is not itself a vulnerability.

import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np
import scipy
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser("Generates a plot of a set of RLLib experiments for the specified metrics.")

    parser.add_argument("experiments", type=str, nargs="*", 
                        help="labels and directories of experiments to plot (label1 dir1 label2 dir2 ...)")
    parser.add_argument("--output", default="mean_return", type=str,
                        help="path to the image file where the plot will be saved")
    parser.add_argument("--x-axis", default="timesteps_total", type=str, 
                        help="column name for x-axis values")
    parser.add_argument("--y-axis", default="episode_reward_mean", type=str, 
                        help="column name for y-axis values")
    parser.add_argument("--x-label", default="time steps", type=str,
                        help="label for the x-axis")
    parser.add_argument("--y-label", default="mean episode return", type=str, 
                        help="label for the y-axis")
    parser.add_argument("--title", default="Mean Episode Return", type=str,
                        help="title for the plot to be generated")
    parser.add_argument("--errors", default="range", type=str,
                        help="error values to plot as shaded regions \{'range', 'deviation', 'error', 'None'\}")
    parser.add_argument("--incomplete", default="truncate", type=str,
                        help="how to handle incomplete runs \{'ignore', 'truncate'\}")

    return parser.parse_args()


def load_experiments(args):
    if len(args) % 2 != 0:
        raise Error("Must provide a label for each experiment")

    experiments = dict()

    for index in range(0, len(args), 2):
        directory = args[index + 1]
        runs = []

        if not os.path.isdir(directory):
            raise Exception(f"Experiment directory {directory} does not exist")

        for obj in os.listdir(directory):
            path = os.path.join(directory, obj)

            if os.path.isdir(path):
                data = pandas.read_csv(os.path.join(path, "progress.csv"))

                # Filter out empy data series
                if data.shape[0] > 0:
                    runs.append(data)
        
        experiments[args[index]] = runs
    
    return experiments


if __name__ == "__main__":
    args = parse_args()

    # Load experiment data
    experiments = load_experiments(args.experiments)

    # Plot results
    color_map = colors.get_cmap("tab20").colors
    legend_entries = []
    y_min = np.infty
    y_max = -np.infty

    plot.clf()

    for index, (label, runs) in enumerate(experiments.items()):
        if len(runs) > 0:

            lengths = [len(run) for run in runs]

            if "ignore" == args.incomplete:
                # Compute the maximum length over runs
                max_length = min(lengths)

                # Remove incomplete sequences
                runs = [run for run in runs if len(run) == max_length]

                # Define x-axis
                x_axis = runs[0][args.x_axis]

                # Construct data series and compute means
                series = [run[args.y_axis] for run in runs]
            else: 
                # Compute the minimum length over runs
                min_length = min(lengths)

                # Define x-axis
                x_axis = runs[0][args.x_axis][0:min_length]

                # Print run information
                print(f"\n\nExperiment: {label}")
                print(f"    data keys: {runs[0].keys()}")

                # Construct data series and compute means
                series = [run[args.y_axis][0:min_length] for run in runs]

            # Convert series data to a single numpy array
            series = np.asarray(series, dtype=np.float32)
            means = np.mean(series, axis=0)

            # Update ranges
            y_min = min(y_min, np.min(series))
            y_max = max(y_max, np.max(series))

            # Compute error bars
            if "range" == args.errors:
                upper = np.max(series, axis=0)
                lower = np.min(series, axis=0)
            elif "deviation" == args.errors:
                std = np.std(series, axis=0, ddof=1)
                upper = means + std
                lower = means - std
            elif "error" == args.errors:
                error = scipy.stats.sem(series, axis=0, ddof=1)
                upper = means + error
                lower = means - error
            else:
                upper = means
                lower = means

            # Plot series
            plot.plot(x_axis, means, color=color_map[2 * index], alpha=1.0)
            plot.fill_between(x_axis, lower, upper, color=color_map[2 * index + 1], alpha=0.3)

        # Add a legend entry even if there were no non-empty data series
        legend_entries.append(patches.Patch(color=color_map[2 * index], label=label))

    # Set ranges
    if y_min > y_max:  # No data, set an arbitrary range
        y_min = 0.0
        y_max = 100.0
    elif 0.0 == y_min and 0.0 == y_max:  # All data is zero, set and arbitrary range
        y_min = -100.0
        y_max = 100.0
    elif y_min > 0.0:  # All values positive, set range from 0 to 120% of max
        y_min = 0.0
        y_max *= 1.2
    elif y_max < 0.0:  # All values negative, set range from 120% of min to 0
        y_min *= 1.2
        y_max = 0.0
    else:  # Both positive and negative values, expand range by 20%
        y_min *= 1.2
        y_max *= 1.2

    # Create plot
    plot.legend(handles=legend_entries)
    plot.title(args.title)
    plot.xlabel(args.x_label)
    plot.ylabel(args.y_label)
    plot.ylim(bottom=y_min, top=y_max)
    plot.savefig(args.output, bbox_inches="tight")
