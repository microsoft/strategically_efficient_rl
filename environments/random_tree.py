import numpy as np

from gym import Env
from gym.spaces import Discrete

# NOTE: What is the difference between the random tree and the tree bandit

class TreeState:

    def __init__(self, index, payoffs=None, successors=None):
        self.index = index
        self.payoffs = payoffs
        self.successors = successors


def build_tree(index, depth, actions, max_depth, bias):
    if 0 >= max_depth - depth:
        payoffs = np.random.beta(1, (1 - bias) / bias, actions)
        return TreeState(index=index, payoffs=payoffs), index + 1
    else:
        successors = []

        for _ in range(actions):
            state, index = build_tree(index=index,
                                      depth=depth + 1,
                                      actions=actions,
                                      max_depth=max_depth,
                                      bias=bias)
            
            successors.append(state)

        return TreeState(index=index,
                         successors=successors), index + 1


class RandomTree(Env):

    def __init__(self, env_config):
        actions = env_config.get("actions", 2)
        depth = env_config.get("depth", 5)
        bias = np.clip(env_config.get("bias", 0.5), 1e-7, 1.)
        
        # Build tree
        self._root, size = build_tree(0, depth=1, actions=actions, max_depth=depth, bias=bias)

        # Build observation and action spaces
        self.observation_space = Discrete(size)
        self.action_space = Discrete(actions)

        # Initialize state
        self._state = None
    
    def reset(self):
        self._state = self._root
        return self._state.index
    
    def step(self, action):

        # Compute reward
        if self._state.payoffs is not None:
            reward = self._state.payoffs[action]
        else:
            reward = 0.0

        # Compute next state
        if self._state.successors is not None:
            self._state = self._state.successors[action]
            done = False
        else:
            done = True

        return self._state.index, reward, done, {}
