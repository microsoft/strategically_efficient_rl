from normal_form.games.zero_sum import ZeroSumGame
import numpy as np


class HybridRoshambo(ZeroSumGame):

    def __init__(self, N, M, config):
        cyclic_actions = config.get("cyclic_actions", 3)
        assert cyclic_actions <= N, "number of cyclic actions must be less than total number of actions"
        assert (cyclic_actions % 2 == 1) or cyclic_actions == 0, "number of cyclic actions must be odd"

        # Save parameters
        self._total_actions = N
        self._cyclic_actions = cyclic_actions

        # Build transitive payoff matrix
        num_actions = N
        transitive_actions = N - cyclic_actions
        G = np.zeros((num_actions, num_actions,))
        
        for i in range(num_actions):
            for j in range(num_actions):
                if i > j:
                    G[i, j] = 1.0
                elif i < j:
                    G[i, j] = 0.0
                else:
                    G[i, j] = 0.5

        # Build cycle payoff matrix
        for i in range(cyclic_actions):
            for j in range(cyclic_actions):
                if i == j:
                    value = 0.5
                elif ((i - j) % cyclic_actions) % 2 == 1:
                    value = 1.0
                else:
                    value = 0.0
                
                G[i + transitive_actions, j + transitive_actions] = value
        
        super(HybridRoshambo, self).__init__(G)

    def __repr__(self):
        return f"hybrid_roshambo_{self._total_actions}_{self._cyclic_actions}"
