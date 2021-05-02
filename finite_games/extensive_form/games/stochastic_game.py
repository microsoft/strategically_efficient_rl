from collections import deque
import numpy as np

from collections import deque
import numpy as np

from extensive_form.games.extensive_game import State, Game


class GameState(State):
    
    def __init__(self, index, depth, player, actions, payoffs=None, successors=None, transitions=None):
        self._index = index
        self._depth = depth
        self._player = player
        self._actions = actions
        self._payoffs = payoffs
        self._successors = successors
        self._transitions = transitions

        self._descendants = set()

        if successors is not None:
            for action in range(actions):
                for state in successors[action]:
                    self._descendants.add(state)

    def _key(self):
        return self._index

    def depth(self):
        return self._depth

    def active_players(self):
        return [self._player]

    def num_actions(self, player_id):
        return self._actions

    def payoffs(self, actions):
        if self._payoffs is not None:
            payoff = self._payoffs[actions[0]]
            return [payoff, 1. - payoff]  # ULCB and Nash-Q ignore the second payoff, but not Nash-V or IQL
        
        return [0., 0.]

    def successors(self, actions):   
        if self._successors is not None:
            return self._successors[actions[0]]
        else:
            return []

    def transitions(self, actions):
        if self._transitions is not None:
            return self._transitions[actions[0]]
        else:
            return []

    def descendants(self):
        if self._descendants is not None:
            return self._descendants
        else:
            return []


class StochasticGame(Game):

    def __init__(self, config):
        self._max_depth = config.get("depth", 5)
        num_states = config.get("states", 5)
        num_actions = config.get("actions", 2)
        num_successors = config.get("successors", 2)
        num_players = config.get("players", 2)
        bias = np.clip(config.get("bias", 0.5), 1e-7, 1.)

        assert 1 == num_players or 2 == num_players, "'players' must be either 1 or 2"
        self._num_players = num_players

        # Build alternating-player rows of states from the final state
        current_row = []
        next_row = None
        index = 0

        transitions = np.ones((num_actions,num_successors)) / num_successors

        for depth in range(self._max_depth, 0, -1):
            player = (depth - 1) % num_players

            for column in range(num_states):
                if self._max_depth <= depth:
                    payoffs = np.random.beta(1, (1 - bias) / bias, num_actions)
                    state = GameState(index, depth, player, num_actions, payoffs=payoffs)
                else:
                    successors = np.random.choice(next_row, size=(num_actions,num_successors), replace=True)
                    state = GameState(index, depth, player, num_actions, 
                                successors=successors, transitions=transitions)

                current_row.append(state)
                index += 1

            next_row = current_row
            current_row = []

        self._initial_states = next_row
        self._initial_distribution = np.ones((num_states,)) / num_states

    def num_players(self):
        return self._num_players

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return 1.

    def initial_states(self):
        return self._initial_states

    def initial_distribution(self):
        return self._initial_distribution
