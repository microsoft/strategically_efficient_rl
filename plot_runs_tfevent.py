#!/usr/bin/env python3

'''
Utility script for generating plots from data stored in RLLib-generated tfevent files
'''

import argparse
import os

import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np
import scipy
import scipy.stats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress unnecessary error messages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

    return parser.parse_args()


def load_experiments(args, x_axis, y_axis):
    if len(args) % 2 != 0:
        raise Error("Must provide a label for each experiment")

    print("\n\n----- Loading Experiments -----")

    experiments = dict()
    x_axis = "ray/tune/" + x_axis
    y_axis = "ray/tune/" + y_axis

    for index in range(0, len(args), 2):
        directory = args[index + 1]
        runs = []

        if not os.path.isdir(directory):
            raise Exception(f"Experiment directory {directory} does not exist")

        for path in os.listdir(directory):
            path = os.path.join(directory, path)

            if os.path.isdir(path):
                for sub_path in os.listdir(path):
                    sub_path = os.path.join(path, sub_path)

                    if os.path.isfile(sub_path) and os.path.basename(sub_path).startswith("events.out.tfevents"):
                        accumulator = EventAccumulator(sub_path)
                        accumulator.Reload()

                        if x_axis in accumulator.scalars.Keys():
                            x_values = [event.value for event in accumulator.Scalars(x_axis)]
                            y_values = [event.value for event in accumulator.Scalars(y_axis)]

                            runs.append((x_values, y_values))

        print(f"Experiment: {args[index]}, {len(runs)} runs")

        if len(runs) > 0:
            experiments[args[index]] = runs

    print("---------------\n")

    return experiments


if __name__ == "__main__":
    args = parse_args()

    # Load experiment data
    experiments = load_experiments(args.experiments, args.x_axis, args.y_axis)

    # Plot results
    color_map = colors.get_cmap("tab20").colors
    legend_entries = []
    y_min = np.infty
    y_max = -np.infty

    plot.clf()

    for index, (label, runs) in enumerate(experiments.items()):
        if len(runs) > 0:

            # Adjust x-axes to match the y-axis, which may not have as many values
            x_axes = []
            y_axes = []

            for run_idx, run in enumerate(runs):
                interval = len(run[0]) // len(run[1])
                x_axes.append(np.asarray(run[0])[(interval - 1)::interval])
                y_axes.append(run[1])

            # Compute minimum run length
            min_length = min([len(y) for y in y_axes])

            # Define x-axis
            x_axis = x_axes[0][0:min_length]

            # Construct data series and compute means
            series = [y[0:min_length] for y in y_axes]

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
