from collections import deque
import numpy as np

from extensive_form.games.extensive_game import State, Game


class TreeState(State):
    
    def __init__(self, index, depth, actions, payoffs=None, successors=None):
        self._index = index
        self._depth = depth
        self._actions = actions
        self._payoffs = payoffs
        self._successors = successors

    def _key(self):
        return self._index

    def depth(self):
        return self._depth

    def active_players(self):
        return [0, 1]

    def num_actions(self, player_id):
        return self._actions

    def payoffs(self, actions):
        if self._payoffs is not None:
            payoff = self._payoffs[actions[0], actions[1]]
            return payoff, 1. - payoff
        
        return 0., 0.

    def successors(self, actions):   
        if self._successors is not None:
            return [self._successors[actions[0], actions[1]]]
        else:
            return []

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        if self._successors is not None:
            descendants = []

            for successors in self._successors:
                for state in successors:
                    descendants.append(state)

            return descendants
        else:
            return []


def build_tree(index, depth, actions, max_depth, bias):
    if 0 >= max_depth - depth:
        payoffs = np.random.beta(1, (1 - bias) / bias, size=(actions, actions))

        return TreeState(index=index,
                         depth=depth,
                         actions=actions,
                         payoffs=payoffs), index + 1

    else:

        # Build intermediate node
        successors = np.empty((actions, actions), dtype=object)

        for a in range(actions):
            for b in range(actions):
                state, index = build_tree(index=index,
                                          depth=depth + 1,
                                          actions=actions,
                                          max_depth=max_depth,
                                          bias=bias)
            
                successors[a,b] = state

        return TreeState(index=index,
                         depth=depth,
                         actions=actions,
                         successors=successors), index + 1


class SimultaneousTreeGame(Game):

    def __init__(self, config):
        self._max_depth = config.get("depth", 4)
        actions = config.get("actions", 2)
        bias = np.clip(config.get("bias", 0.5), 1e-7, 1.)

        self._root, _ = build_tree(index=0, 
                                   depth=1,
                                   actions=actions,
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
                row_action = np.random.choice(actions)

                if state._successors is None:
                    for column_action in range(actions):
                        state._payoffs[row_action, column_action] = row_value
                else:
                    for column_action in range(actions):
                        states_to_process.append(state._successors[row_action, column_action])

    def num_players(self):
        return 2

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return 1.

    def initial_states(self):
        return [self._root]

    def initial_distribution(self):
        return [1.]
