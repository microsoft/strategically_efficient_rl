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
            return payoff, 1. - payoff
        
        return 0., 0.

    def successors(self, actions):
        if self._successors is not None:
            return [self._successors[actions[0]]]
        
        return []

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        if self._successors is not None:
            return self._successors
        else:
            return []


def build_tree(index, depth, player, actions, max_depth, payoff):
    if 0 >= max_depth - depth:
        payoffs = np.full(actions, payoff, dtype=np.float64)

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
                                      payoff=payoff)
            
            successors.append(state)

        return TreeState(index=index,
                         depth=depth,
                         player=player,
                         actions=actions,
                         successors=successors), index + 1


class DecoyTree(Game):

    def __init__(self, config):
        target_depth = config.get("target_depth", 5)
        target_actions = config.get("target_actions", 2)
        target_payoff = config.get("target_payoff", 0)

        decoy_trees = config.get("decoy_trees", 1)
        decoy_depth = config.get("decoy_depth", 5)
        decoy_actions = config.get("decoy_actions", 2)
        decoy_payoff = config.get("decoy_payoff", 1)

        # Build decoy trees
        decoy_roots = []
        index = 0

        for _ in range(decoy_trees):
            decoy_root, index = build_tree(index=index,
                                           depth=3,
                                           player=0,
                                           actions=decoy_actions,
                                           max_depth=decoy_depth + 2,
                                           payoff=decoy_payoff)

            decoy_roots.append(decoy_root)

        # Build adversary nodes for decoy trees
        root_nodes = []

        for decoy_root in decoy_roots:
            failure_node = TreeState(index=index,
                                     depth=3,
                                     player=0,
                                     actions=1,
                                     payoffs=[0.])
            
            root_nodes.append(TreeState(index=index + 1,
                                        depth=2,
                                        player=1,
                                        actions=2,
                                        successors=[decoy_root, failure_node]))
            index += 2

        # Build target tree
        target_root, index = build_tree(index=index, 
                                        depth=2,
                                        player=1,
                                        actions=target_actions,
                                        max_depth=target_depth + 1,
                                        payoff=target_payoff)
        
        root_nodes.append(target_root)

        # Make sure the row_player can force a tie in the target tree
        states_to_process = deque()
        states_to_process.append(target_root)

        while len(states_to_process) > 0:
            state = states_to_process.popleft()

            if 0 == state._player:
                action = np.random.choice(state._actions)

                if state._successors is None:
                    state._payoffs[action] = 0.5
                else:
                    states_to_process.append(state._successors[action])
                
            else:

                if state._successors is None:
                    state._payoffs = np.full(state._actions, 0.5)
                else:
                    states_to_process.extend(state._successors)

        # Build root node
        self._root = TreeState(index=index, 
                               depth=1,
                               player=0,
                               actions=decoy_trees + 1, 
                               successors=root_nodes)

    def num_players(self):
        return 2

    def max_payoff(self):
        return 1.

    def initial_states(self):
        return [self._root]

    def initial_distribution(self):
        return [1.]
