from normal_form.games.zero_sum import ZeroSumGame
import numpy as np


class Roshambo(ZeroSumGame):

    def __init__(self, N, M, config):
        assert N % 2 == 1, "Must have an odd number of actions"

        G = np.zeros((N, N,))

        for i in range(N):
            for j in range(N):
                if i == j:
                    G[i, j] = 0.5
                elif ((i - j) % N) % 2 == 1:
                    G[i, j] = 1.0
                else:
                    G[i, j] = 0.0
        
        super(Roshambo, self).__init__(G)
    
    def __repr__(self):
        return f"roshambo_{self.G.shape[0]}"
