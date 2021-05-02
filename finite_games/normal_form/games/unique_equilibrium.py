from normal_form.games.zero_sum import ZeroSumGame
import numpy as np


class UniqueEquilibrium(ZeroSumGame):

    def __init__(self, N, M, config):
        G = np.zeros((N, M))
        row = np.random.choice(N)
        column = np.random.choice(M)

        G[row, column] = 0.5

        for i in range(M):
            if i != column:
                G[row, i] = 0.0
        
        for i in range(N):
            if i != row:
                for j in range(M):
                    G[i, j] = 1.0

        super(UniqueEquilibrium, self).__init__(G)

    def __repr__(self):
        return f"unique_{self.G.shape[0]}_{self.G.shape[1]}"
