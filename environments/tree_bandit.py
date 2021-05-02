import numpy as np

from gym import Env
from gym.spaces import Discrete


class TreeBandit(Env):

    def __init__(self, env_config):
        self._max_depth = env_config.get("depth", 4)
        self._base_reward = env_config.get("base", 1.0)
        self._bonus_reward = env_config.get("bonus", 2.0)
        self._noise = env_config.get("noise", 5.0)
        self._scramble = env_config.get("scramble", False)

        self._num_states = 2**(self._max_depth + 1) - 1

        self.observation_space = Discrete(self._num_states)
        self.action_space = Discrete(2)

        self._depth = 0
        self._state = 0

    def reset(self):
        self._depth = 0
        self._index = 0
        self._state = 0

        return self._state

    def step(self, action):
        
        # Translate action - The index of the left/right actions should vary
        if self._scramble:
            action = (self._state + action) % 2

        # Update state
        if self._depth < self._max_depth:
            self._depth += 1
            self._index = (self._index * 2) + action
            self._state = self._index + (2**self._depth) - 1 
        
        # Compute reward
        if self._depth == self._max_depth:
            reward = np.random.normal(self._base_reward, self._noise)

            if 0 == self._index:
                reward += self._bonus_reward
        else:
            reward = 0.0

        # Determine if we are done
        done = (self._depth == self._max_depth)

        return self._state, reward, done, {}
