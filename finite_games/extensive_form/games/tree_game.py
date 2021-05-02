'''
Defines a tree-structured, alternating move game with uniformly random payoffs (in [0,1]) for the first player.
'''

from collections import deque
import numpy as np

from extensive_form.games.extensive_game import State, Game


class TreeState(State):
    
    def __init__(self, index, depth, player, actions, payoffs=None, successors=None):
        self._index = index
        self._depth = depth
        self._player = player
        self._actions = actions
        self._payoffs = payoffs
        self._successors = successors

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
            payoff = self._payoffs[actions[0]]  # Do we have enough payoffs for the number of actions? - yes
            return [payoff, 1. - payoff]
        
        return [0., 0.]  # This is wrong - not really constant sum - second payoff ignored

    def successors(self, actions):   
        if self._successors is not None:
            return [self._successors[actions[0]]]
        else:
            return []

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        if self._successors is not None:
            return self._successors
        else:
            return []


def build_tree(index, depth, player, actions, num_players, max_depth, bias):
    if 0 >= max_depth - depth:
        payoffs = np.random.beta(1, (1 - bias) / bias, actions)

        return TreeState(index=index,
                         depth=depth,
                         player=player,
                         actions=actions,
                         payoffs=payoffs), index + 1

    else:

        # Build intermediate node
        next_player = (player + 1) % num_players
        successors = []

        for _ in range(actions):
            state, index = build_tree(index=index,
                               depth=depth + 1,
                               player=next_player, 
                               actions=actions,
                               num_players=num_players,
                               max_depth=max_depth,
                               bias=bias)
            
            successors.append(state)

        return TreeState(index=index,
                         depth=depth,
                         player=player,
                         actions=actions,
                         successors=successors), index + 1


class TreeGame(Game):

    def __init__(self, config):
        self._max_depth = config.get("depth", 4)
        actions = config.get("actions", 2)
        num_players = config.get("players", 2)
        bias = np.clip(config.get("bias", 0.5), 1e-7, 1.)

        assert 1 == num_players or 2 == num_players, "'players' must be either 1 or 2"
        self._num_players = num_players

        self._root, _ = build_tree(index=0, 
                                   depth=1,
                                   player=0, 
                                   actions=actions,
                                   num_players=num_players,
                                   max_depth=self._max_depth, 
                                   bias=bias)

        # Allow the row player to guarantee a certain payoff if requested
        if "row_value" in config:
            row_value = np.clip(config["row_value"], 0., 1.)

            # Make sure the row_player can force a tie
            states_to_process = deque()
            states_to_process.append(self._root)

            while len(states_to_process) > 0:
                state = states_to_process.popleft()

                if 0 == state._player:
                    action = np.random.choice(state._actions)

                    if state._successors is None:
                        state._payoffs[action] = row_value
                    else:
                        states_to_process.append(state._successors[action])
                else:

                    if state._successors is None:
                        state._payoffs = np.full(state._actions, row_value)
                    else:
                        states_to_process.extend(state._successors)

    def num_players(self):
        return self._num_players

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return 1.

    def initial_states(self):
        return [self._root]

    def initial_distribution(self):
        return [1.]
