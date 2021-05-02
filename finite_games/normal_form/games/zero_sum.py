import numpy as np


class ZeroSumGame:

    def __init__(self, G):
        self.G = G
        self.N = G.shape[0]
        self.M = G.shape[1]

    def sample(self, row, column):
        payoff = self.G[row, column]
        return payoff, 1.0 - payoff
    
    def nash_conv(self, row, column):
        row_payoffs = self.G.dot(column)
        column_payoffs = row.dot(self.G)

        row_payoff = np.min(column_payoffs)
        column_payoff = 1.0 - np.max(row_payoffs)

        nash_conv = np.max(row_payoffs) - np.min(column_payoffs)

        return row_payoff, column_payoff, nash_conv 


class RandomZeroSumGame(ZeroSumGame):

    def __init__(self, N, M, config):
        super(RandomZeroSumGame, self).__init__(np.random.random((N, M,)))

    def __repr__(self):
        return f"zero_sum_{self.G.shape[0]}_{self.G.shape[1]}"
