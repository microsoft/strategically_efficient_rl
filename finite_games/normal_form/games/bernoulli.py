from normal_form.games.zero_sum import ZeroSumGame
import numpy as np


class BernoulliGame(ZeroSumGame):

    def __init__(self, G):
        super(BernoulliGame, self).__init__(G)

    def sample(self, row, column):
        win = np.random.binomial(1, self.G[row, column])

        return win, 1.0 - win


class RandomBernoulliGame(BernoulliGame):

    def __init__(self, N, M, config):
        super(RandomBernoulliGame, self).__init__(np.random.random((N, M,)))
    
    def __repr__(self):
        return f"bernoulli_{self.G.shape[0]}_{self.G.shape[1]}"
