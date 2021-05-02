'''
Defines a tree-structured, alternating move game with random binary (0 or 1) payoffs for the first player.
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
            payoff = self._payoffs[actions[0]]
            return [payoff, 1. - payoff]
        
        return [0.0, 0.0]

    def successors(self, actions):   
        if self._successors is not None:
            return [self._successors[actions[0]]]
        else:
            return []

    def transitions(self, actions):
        return [1.0]

    def descendants(self):
        if self._successors is not None:
            return self._successors
        else:
            return []


def build_tree(index, depth, player, actions, max_depth, bias):
    if 0 >= max_depth - depth:
        payoffs = np.random.binomial(1, bias, size=(actions,))

        return TreeState(index=index,
                         depth=depth,
                         player=player,
                         actions=actions,
                         payoffs=payoffs), index + 1

    else:

        # Build intermediate node
        next_player = (player + 1) % 2
        successors = []

        for _ in range(actions):
            state, index = build_tree(index=index,
                               depth=depth + 1,
                               player=next_player, 
                               actions=actions,
                               max_depth=max_depth,
                               bias=bias)
            
            successors.append(state)

        return TreeState(index=index,
                         depth=depth,
                         player=player,
                         actions=actions,
                         successors=successors), index + 1


class AlphaBeta(Game):

    def __init__(self, config):
        self._max_depth = config.get("depth", 4)
        actions = config.get("actions", 2)
        bias = np.clip(config.get("bias", 0.5), 0.0, 1.0)

        self._root, _ = build_tree(index=0, 
                                   depth=1,
                                   player=0, 
                                   actions=actions,
                                   max_depth=self._max_depth, 
                                   bias=bias)

    def num_players(self):
        return 2

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return 1.0

    def initial_states(self):
        return [self._root]

    def initial_distribution(self):
        return [1.0]
