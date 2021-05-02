import numpy as np


class Hedge:

    def __init__(self, config):
        self._epsilon = config.get("epsilon", 0.01)

    def __call__(self, row_payoffs, column_payoffs=None):
        if column_payoffs is None:
            row_payoffs = -column_payoffs

        G = row_payoffs
        GP = column_payoffs.T

        N, M = G.shape

        steps = np.ceil(np.log(max(N, M)) / (2 * self._epsilon**2))
        row_scale = np.sqrt(8 * np.log(N) / steps)
        column_scale = np.sqrt(8 * np.log(M) / steps)

        row_strategy = np.ones(N)
        column_strategy = np.ones(M)

        row_strategies = np.zeros(N)
        column_strategies = np.zeros(M)

        for _ in range(int(steps)):
            row_strategy /= np.sum(row_strategy)
            column_strategy /= np.sum(column_strategy)

            row_strategies += row_strategy
            column_strategies += column_strategy

            row_logits = row_scale * G.dot(column_strategy)
            column_logits = column_scale * GP.dot(row_strategy)

            row_strategy *= np.exp(row_logits - np.max(row_logits))
            column_strategy *= np.exp(column_logits - np.max(column_logits))

        row_strategies /= np.sum(row_strategies)
        column_strategies /= np.sum(column_strategies)

        return row_strategies, column_strategies, {}

    def __repr__(self):
        return "hedge"