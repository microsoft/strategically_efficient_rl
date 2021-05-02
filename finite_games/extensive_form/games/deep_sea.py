import numpy as np

from extensive_form.games.extensive_game import State, Game


class DeepSeaState(State):

    def __init__(self, index, depth, payoffs, successors):
        self._index = index
        self._depth = depth
        self._payoffs = payoffs
        self._successors = successors

        if np.random.random() <= 0.5:
            self._action_map = (0, 1)
        else:
            self._action_map = (1, 0)

    def _key(self):
        return self._index

    def depth(self):
        return self._depth

    def active_players(self):
        return [0]

    def num_actions(self, player_id):
        return 2

    def payoffs(self, actions):
        return self._payoffs[self._action_map[actions[0]]]

    def successors(self, actions): 
        if self._successors is None:
            return []
        else:
            return [self._successors[self._action_map[actions[0]]]]

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        if self._successors is None:
            return []
        else:
            return self._successors


class DeepSea(Game):

    def __init__(self, config={}):
        
        # Get configuration
        size = config.get("size", 10)
        penalty = config.get("penalty", 0.01)

        # Build last layer of game - this is where we get the big payoff if we reach the goal
        last_layer = []
        index = 0

        for _ in range(size - 1):
            last_layer.append(DeepSeaState(index, size, payoffs=[[0., 1.], [0., 1.]], successors=None))
            index += 1

        last_layer.append(DeepSeaState(index, size, payoffs=[[1., 0.], [1., 0.]], successors=None))
        index += 1

        # Build intermediate layers - in reverse depth order
        payoffs = [[penalty, 0.], [0., penalty]]

        for depth in range(size - 1, 0, -1):
            layer = []

            for idx in range(size):
                left= last_layer[max(idx - 1, 0)]
                right = last_layer[min(idx + 1, size - 1)]

                layer.append(DeepSeaState(index, depth, payoffs=payoffs, successors=[left, right]))
                index += 1

            last_layer = layer
        
        # Define initial states
        self._initial_states = [last_layer[0]]

        # Compute maximum payoff and depth
        self._max_payoff = 1. + (size - 1) * penalty
        self._max_depth = size

    def num_players(self):
        return 2

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return self._max_payoff

    def initial_states(self):
        return self._initial_states

    def initial_distribution(self):
        return [1.]
