"""
Defines the root class for all meta-strategies.  Also maintains a registry of meta-strategies, and a method for 
"""

from abc import ABC, abstractmethod

from algorithms.psro.self_play import SelfPlay
from algorithms.psro.fictitious_play import FictitiousPlay


class EvaluationProfile:
    """
    Represents a single pure-strategy profile for which we want a set of samples, and a handle for storing those samples.
    """

    def __init__(self, policy_ids, policy_epochs, num_samples):
        self.policy_ids = policy_ids
        self.policy_epochs = policy_epochs
        self.num_samples = num_samples

        # Set of sample payoff vectors
        self._samples = []

    def get_mapping_fn(self):
        return lambda idx: self.policy_ids[idx]

    def add_sample(self, payoffs):
        self._samples.append(payoffs)


class MetaStrategy(ABC):

    def __init__(self, num_players, is_symmetric, config):  # How will this handle groups in RLLib
        self.num_players = num_players
        self.is_symmetric = is_symmetric
        self.config = config

        # Validate shape of array of policy IDS
        if self.is_symmetric:
            self.policy_ids = []
        else:
            self.policy_ids = [[] for _ in range(num_players)]

    def _on_add_policy(self, policy_id, player):
        pass

    def _on_add_policy_symmetric(self, policy_id):
        pass

    def add_policy(self, policy_id, player):
        if self.is_symmetric:
            self.policy_ids.append(policy_id)
            self._on_add_policy_symmetric(policy_id)
        else:
            self.policy_ids[player].append(policy_id)
            self._on_add_policy(policy_id, player)

    @abstractmethod
    def _get_mapping_fn(self, learning_player):
        pass

    @abstractmethod
    def _get_mapping_fn_symmetric(self, learning_player):
        pass

    @abstractmethod
    def get_mapping_fn(self, learning_player):  # Should randomize over opponents, but leaves the learning player alone
        if self.is_symmetric:
            if len(self.policy_ids) < 2:
                raise Exception("For symmetric games, at least two policies must be defined before training")

            return self._get_mapping_fn_symmetric(learning_player)
        else:
            if any([len(ids) < 1 for ids in self.policy_ids]):
                raise Exception("For asymmetric games, each player must have at least one policy defined before training")

            return self._get_mapping_fn(learning_player)

    def request_evaluation_profiles(self):
        return
        yield


METASTRATEGIES = {
    "self_play": SelfPlay,
    "fictitious_play": FictitiousPlay,
}


def make_metastrategy(name, num_players, is_symmetric, config):
    if name not in METASTRATEGIES:
        raise ValueError(f"Meta-strategy '{name}' is unknown")

    cls = METASTRATEGIES[name]
    return cls(num_players, is_symmetric, config)