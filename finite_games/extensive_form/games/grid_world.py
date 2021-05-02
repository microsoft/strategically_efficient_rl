import numpy as np

from extensive_form.games.extensive_game import State, Game

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


class GridState(State):

    def __init__(self, index, depth, row, column, payoff, successors):
        self._index = index
        self._depth = depth
        self._row = row
        self._column = column
        self._successors = successors
        self._payoff = payoff

    def _key(self):
        return self._index

    def depth(self):
        return self._depth

    def active_players(self):
        return [0]

    def num_actions(self, player_id):
        return 5

    def payoffs(self, actions):
        return [self._payoff, 1.0 - self._payoff]

    def successors(self, actions):
        if self._successors is not None:
            return [self._successors[actions[0]]]
        
        return []

    def transitions(self, actions):
        return [1.0]

    def descendants(self):
        if self._successors is not None:
            return self._successors
        else:
            return []


def build_layer(index, depth, width, height, cost, goals, next_layer=None):
    layer = np.empty((width, height), dtype=np.object)

    for row in range(width):
        for column in range(height):
            if next_layer is not None:
                successors = [next_layer[row, column]] * 5

                if row > 0:
                    successors[UP] = next_layer[row - 1, column]
                if row + 1 < height:
                    successors[DOWN] = next_layer[row + 1, column]
                if column > 0:
                    successors[LEFT] = next_layer[row, column - 1]
                if column + 1 < width:
                    successors[RIGHT] = next_layer[row, column + 1]
            else:
                successors = None
            
            if (row, column) in goals:
                payoff = 1.0
            else:
                payoff = 1.0 - cost

            layer[row, column] = GridState(index, depth, row, column, payoff, successors)
            index += 1
    
    return layer, index


class GridWorld(Game):

    def __init__(self, config):
        cost = config.get("cost", 0.1)
        width = config.get("width", 5)
        height = config.get("height", 5)

        self._max_depth = config.get("depth", width + height)

        # Generate random goal if needed
        if "goals" in config:
            goals = config["goal"]
        else:
            goal_row = np.random.randint(0, height)
            goal_column = np.random.randint(0, width)
            goals = [(goal_row, goal_column)]

        # Build layers - in reverse order
        layer = None
        index = 0

        for depth in range(self._max_depth, 0, -1):
            layer, index = build_layer(index, depth, width, height, cost, goals, layer)

        self._initial_states = list(layer.flatten())
        self._initial_distribution = np.ones(len(self._initial_states))
        self._initial_distribution /= np.sum(self._initial_distribution)

    def num_players(self):
        return 1

    def max_depth(self):
        return self._max_depth

    def max_payoff(self):
        return 1.0 * self._max_depth

    def initial_states(self):
        return self._initial_states

    def initial_distribution(self):
        return self._initial_distribution
