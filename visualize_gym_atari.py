#!/usr/bin/env python3

'''
Script for visualizing learned policies in the gym Atari environment.
'''

import argparse
from collections import defaultdict
import gym
import importlib
import numpy as np
import os
import pickle
import time

import ray
from ray.tune.registry import get_trainable_cls
from supersuit import frame_stack_v1, color_reduction_v0, resize_v0, normalize_obs_v0, dtype_v0

import algorithms
import environments


class RLLibPolicy:
    """ Represents a single policy contained in an RLLib Trainer """

    def __init__(self, trainer, policy_id):
        self._trainer = trainer
        self._policy_id = policy_id

        # Get the local policy object for the given ID
        policy = trainer.get_policy(policy_id)
        
        # Sample an action from the action space for this policy to act as the previous action for the first step
        self._initial_action = 0
        
        # Get the initial state for a recurrent policy if needed
        initial_rnn_state = policy.get_initial_state()
        
        if initial_rnn_state is not None and len(initial_rnn_state) > 0:
            self._initial_rnn_state = initial_rnn_state
        else:
            self._initial_rnn_state = None

        # Initialize the policy - only affects the wrapper, not the underlying policy
        self.reset()

    def reset(self):
        self._prev_action = self._initial_action
        self._prev_rnn_state = self._initial_rnn_state

    def action(self, obs, prev_reward=0.0):
        if self._initial_rnn_state is not None:
            self._prev_action, self._prev_state, _ = self._trainer.compute_action(obs,
                                                        state=self._prev_rnn_state,  
                                                        prev_action=self._prev_action,
                                                        prev_reward=prev_reward,
                                                        policy_id=self._policy_id)
        else:
            self._prev_action = self._trainer.compute_action(obs,
                                    prev_action=self._prev_action,
                                    prev_reward=prev_reward,
                                    policy_id=self._policy_id)

        return self._prev_action


def load_last_checkpoint(run, trainer_cls):

    # Build trainable with appropriate configuration
    with open(os.path.join(run, "params.pkl"), "rb") as f:
        config = pickle.load(f)

    config["num_workers"] = 0
    config["num_gpus"] = 0

    # Because RLLib is stupid, a log directory is required even when we are using a NoopLogger
    trainer = trainer_cls(config=config)

    # Get checkpoint IDs
    checkpoint_ids = []

    for obj in os.listdir(run):
        if obj.startswith("checkpoint_"):
            checkpoint_ids.append(int(obj[11:]))
    
    checkpoint_ids.sort()

    # Load final checkpoint
    checkpoint = str(checkpoint_ids[-1])
    
    # Don't restore, see if this affects the trainer config
    checkpoint = os.path.join(run, "checkpoint_" + checkpoint, "checkpoint-" + checkpoint)
    trainer.restore(checkpoint)

    return trainer, config


def parse_args():
    parser = argparse.ArgumentParser("Visualizes a set of trained policies in the Gym Atari environments")

    parser.add_argument("trial", type=str, 
                        help="path to the training run to visualize")
    parser.add_argument("-a", "--alg", type=str, default="PPO",
                        help="name of the Trainable class from which the checkpoints were generated")
    parser.add_argument("-e", "--episodes", type=int, default=20,
                        help="the number of episodes to roll out")

    return parser.parse_args()


def main(args):

    # Initialize ray
    ray.init(num_cpus=1)

    # Get trainable class
    trainer_cls = get_trainable_cls(args.alg)
    trainer, config = load_last_checkpoint(args.trial, trainer_cls)

    # Get environment config
    env_config = config["env_config"].copy()
    env_name = env_config.pop("game")
    frame_stack = env_config.pop("frame_stack", 4)
    color_reduction = env_config.pop("color_reduction", True)

    agent_id = env_config.pop("agent_id", "first_0")

    # Build Atari environment
    env = gym.make(env_name)

    # Wrap environment in preprocessors
    wrapped_env = env

    if color_reduction:
        wrapped_env = color_reduction_v0(wrapped_env, mode='full')

    wrapped_env = resize_v0(wrapped_env, 84, 84)
    wrapped_env = dtype_v0(wrapped_env, dtype=np.float32)
    wrapped_env = frame_stack_v1(wrapped_env, frame_stack)
    wrapped_env = normalize_obs_v0(wrapped_env, env_min=0, env_max=1)

    # Reset environment
    obs = wrapped_env.reset()
    env.render()

    # Initialize policies
    policy = RLLibPolicy(trainer, f"policy_{agent_id}")
    
    # Roll-out and visualize policies
    step = 0
    episode = 0

    while episode < args.episodes:

        print(f"episode {episode}, step {step}")
        step += 1
        
        action = policy.action(obs)
        obs, reward, done, info = wrapped_env.step(action)

        # Render environment
        env.render()
        time.sleep(0.01)

        # Reset if necessary
        if done:
            obs = wrapped_env.reset()
            episode += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
