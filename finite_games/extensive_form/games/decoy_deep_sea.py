import numpy as np

from extensive_form.games.extensive_game import State, Game, Environment


class DeepSeaState(State):

    def __init__(self, index, depth, player, actions, action_map=None, payoffs=None, successors=None):
        self._index = index
        self._depth = depth
        self._player = player
        self._actions = actions

        self._action_map = action_map or np.arange(actions)

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
        if self._payoffs is None:
            return [0., 0.]
        else:
            return self._payoffs[self._action_map[actions[0]]]

    def successors(self, actions): 
        if self._successors is None:
            return []

        successor = self._successors[self._action_map[actions[0]]]
        
        if successor is None:
            return []
        else:
            return [successor]

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        if self._successors is None:
            return []
        else:
            return [s for s in self._successors if s is not None]


def build_deep_sea(index, depth, size, penalty, payoff):

    # Build last layer of game - this is where we get the big payoff if we reach the goal
    final_depth = depth + size - 1
    last_layer = []

    for _ in range(size - 1):
        last_layer.append(DeepSeaState(index=index, 
                                       depth=final_depth, 
                                       player=0,
                                       actions=1,
                                       payoffs=[[1. - payoff, payoff]]))
        index += 1

    last_layer.append(DeepSeaState(index=index, 
                                   depth=final_depth,
                                   player=0,
                                   actions=1,
                                   payoffs=[[payoff, 1. - payoff]]))
    index += 1

    # Build intermediate layers - in reverse depth order
    for layer_depth in range(final_depth - 1, depth - 1, -1):
        layer = []

        for idx in range(size):
            left= last_layer[max(idx - 1, 0)]
            right = last_layer[min(idx + 1, size - 1)]

            layer.append(DeepSeaState(index=index, 
                                      depth=layer_depth, 
                                      player=0,
                                      actions=2,
                                      payoffs=[[penalty, 0.], [0., penalty]], 
                                      successors=[left, right]))
            index += 1

        last_layer = layer
    
    # Define initial states
    return last_layer[0], index


class DecoyDeepSea(Game):

    def __init__(self, config={}):
        
        # Get configuration
        decoy_trees = config.get("decoy_games", 1)
        decoy_size = config.get("decoy_size", 5)
        decoy_payoff = config.get("decoy_payoff", 1.)

        adversary_payoff = config.get("adversary_payoff", 1.0)

        target_size = config.get("target_size", decoy_size)
        target_payoff = config.get("target_payoff", 1.)

        target_penalty = config.get("penalty", 0.)

        # Build decoy games
        decoy_penalty = target_penalty *  (target_size - 1) / (decoy_size - 1)

        decoy_roots = []
        index = 0

        for _ in range(decoy_trees):
            decoy_root, index = build_deep_sea(index=index,
                                               depth=3,
                                               size=decoy_size,
                                               penalty=decoy_penalty,
                                               payoff=decoy_payoff)

            decoy_roots.append(decoy_root)

        # Build adversary nodes for decoy trees
        root_nodes = []

        for decoy_root in decoy_roots:
            root_nodes.append(DeepSeaState(index=index,
                                           depth=2,
                                           player=1,
                                           actions=2,
                                           payoffs=[[0., 0.], [1. - adversary_payoff, adversary_payoff]],
                                           successors=[decoy_root, None]))
            index += 1

        # Build target tree
        target_root, index = build_deep_sea(index=index,
                                            depth=2,
                                            size=target_size,
                                            penalty=target_penalty,
                                            payoff=target_payoff)
        
        self._target_action = np.random.randint(0, len(root_nodes))
        root_nodes.insert(self._target_action, target_root)

        # Build root node
        self._root = DeepSeaState(index=index, 
                                  depth=1,
                                  player=0,
                                  actions=decoy_trees + 1, 
                                  successors=root_nodes)
        
        # Compute maximum payoff and depth
        self._max_payoff = 1. + (target_size - 1) * target_penalty
        self._max_depth = max(decoy_size, target_size) + 2

    def num_players(self):
        return 2

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return self._max_payoff

    def initial_states(self):
        return [self._root]

    def initial_distribution(self):
        return [1.]

    def build_env(self):
        return DecoyEnvironment(self)


class DecoyEnvironment(Environment):

    def __init__(self, game):
        super(DecoyEnvironment, self).__init__(game)

    def get_stats(self, state, actions):
        if "target_count" not in self._stats:
            self._stats["target_count"] = 0
            self._stats["decoy_count"] = 0

        if self._game._root == state:
            if actions[0] == self._game._target_action:
                self._stats["target_count"] += 1
            else:
                self._stats["decoy_count"] += 1
