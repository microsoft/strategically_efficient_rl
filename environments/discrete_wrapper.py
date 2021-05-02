import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box


class DiscreteWrapper(ObservationWrapper):
    """ Wraps a discrete environment in one which retruns one-hot encoded observations. """

    def __init__(self, env):
        super(DiscreteWrapper, self).__init__(env)
        self.observation_space = Box(-1.0, 1.0, shape=(env.observation_space.n,))

    def observation(self, state):
        obs = np.zeros(self.observation_space.shape)
        obs[state] = 1.0
        
        return obs
