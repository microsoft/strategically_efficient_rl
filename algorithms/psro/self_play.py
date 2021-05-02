from algorithms.psro.metastrategy import MetaStrategy


class SelfPlay(MetaStrategy):

    def __init__(self, num_players, is_symmetric, config):
        super(SelfPlay, self).__init__(num_players, is_symmetric, config)

    def _get_mapping_fn(self, learning_player):
        return lambda idx: self.policy_ids[idx][-1]

    def _get_mapping_fn_symmetric(self, learning_player):
        return lambda idx: self.policy_ids[(-1 if idx == learning_player else -2)]
