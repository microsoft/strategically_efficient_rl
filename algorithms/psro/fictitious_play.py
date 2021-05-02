import numpy as np

from algorithms.psro.metastrategy import MetaStrategy


class FictitiousPlay(MetaStrategy):

    def __init__(self, num_players, is_symmetric, config):
        super(FictitiousPlay, self).__init__(num_players, is_symmetric, config)

        if self.is_symmetric:
            self._weights = []
        else:
            self._weights = [[] for _ in range(num_players)]

        self._decay = self.config.get("decay", 1.0)
        
        if not 0.0 < self._decay <=1.0:
            raise Exception("Probability decay constant must be in (0,1]")

    def _on_add_policy(self, policy_id, player):
        if len(self._weights[player]) > 0: 
            self._weights[player] = (self._decay * np.asarray(self._weights[player])).tolist()

        self._weights[player].append(1.0)

    def _on_add_policy_symmetric(self, policy_id):
        if len(self._weights) > 0: 
            self._weights[-1] = 1.0
            self._weights = (self._decay * np.asarray(self._weights[player])).tolist()

        self._weights[player].append(0.0)

    def _normalize(self, weights):
        return np.asarray(weights) / sum(weights)

    def _get_mapping_fn(self, learning_player):
        distribution = [self._normalize(weights) for weights in self._weights]

        def policy_mapping_fn(idx):
            if idx == learning_player:
                return self.policy_ids[learning_player][-1]
            else:
                return np.random.choice(self.policy_ids[learning_player], p=distribution[learning_player])

        return policy_mapping_fn

    def _get_mapping_fn_symmetric(self, learning_player):
        distribution = self._normalize(self._weights)

        def policy_mapping_fn(idx):
            if idx == learning_player:
                return self.policy_ids[-1]
            else:
                return np.random.choice(self.policy_ids, p=distribution)

        return policy_mapping_fn
