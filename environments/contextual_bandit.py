from collections import namedtuple
import numpy as np

from gym import Env
from gym.spaces import Box, Discrete


Context = namedtuple("Context", ["observation", "rewards"])


def matching(num_features, num_actions, config):
    contexts = []

    for feature in range(num_features):
        observation = np.zeros(num_features)
        observation[feature] = 1.0

        rewards = np.zeros(num_actions)
        rewards[feature % num_actions] = 1.0 

        contexts.append(Context(observation, rewards))

    return contexts


def counting(num_features, num_actions, config):
    num_instances = config.get("instances", 5)
    contexts = []

    for action in range(num_actions):
        for instance in range(instances):
            indices = np.random.choice(num_features, size=action, replace=False)
            observation = np.zeros(num_features)

            for idx in indices:
                observation[idx] = 1.0
            
            rewards = np.zeros(num_actions)
            rewards[action] = 1.0

            contexts.append(Context(observation, rewards)) 


TASKS = {
    "matching": matching,
    "counting": counting,
}


class ContextualBandit(Env):

    def __init__(self, env_config):
        num_features = env_config.pop("num_features", 10)
        num_actions = env_config.pop("num_actions", 5)
        task_name = env_config.pop("task", "matching")

        if task_name not in TASKS:
            raise ValueError(f"Contextual bandit task '{task_name}' is undefined")

        self._contexts = TASKS[task_name](num_features, num_actions, env_config)
        self._current_context = self._contexts[np.random.randint(0, len(self._contexts))]

        self.observation_space = Box(0.0, 1.0, (num_features,))
        self.action_space = Discrete(num_actions)

    def reset(self):
        self._current_context = self._contexts[np.random.randint(0, len(self._contexts))]
        return self._current_context.observation

    def step(self, action):
        reward = self._current_context.rewards[action]
        return self._current_context.observation, reward, True, {}
