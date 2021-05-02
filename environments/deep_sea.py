import numpy as np

from gym import Env
from gym.spaces import Discrete


class DeepSea(Env):

    def __init__(self, env_config):
        self._size = env_config.get("size", 10)
        self._penalty = env_config.get("penalty", 0.01)
        self._reward = env_config.get("reward", 1.0)
        self._noise = env_config.get("noise", 0.0)
        self._scramble = env_config.get("scramble", False)

        self.observation_space = Discrete(self._size**2)
        self.action_space = Discrete(2)

        self._max_steps = self._size - 1
        self._num_steps = 0
        self._row = 0
        self._column = 0

    def _state(self):
        return (self._size * self._row) + self._column

    def reset(self):
        self._num_steps = 0
        self._row = 0
        self._column = 0

        return self._state()

    def step(self, action):

        # Translate action - The index of the left/right actions should vary
        if self._scramble:
            action = (self._row + self._column + action) % 2
        
        # Update position and compute reward
        reward = 0.0

        if 0 == action:
            self._column = max(0, self._column - 1)
        else:
            self._column = min(self._max_steps, self._column + 1)
            reward -= self._penalty / self._size

        self._row = min(self._max_steps, self._row + 1)

        if self._max_steps == self._column:
            reward += self._reward

            if 0.0 != self._noise:
                reward += np.random.random(scale=self._noise)

        # Determine if we are done
        self._num_steps += 1
        done = (self._max_steps <= self._num_steps)

        # Encode state as an integer and return results
        return self._state(), reward, done, {}
