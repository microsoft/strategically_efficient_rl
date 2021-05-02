from extensive_form.games.extensive_game import State, Game
from normal_form.games import build_game


class RepeatedGameState(State):

    def __init__(self, G, depth, max_depth):
        self._G = G
        self._depth = depth
        self._successors = []

        if depth < max_depth:
            self._successors.append(RepeatedGameState(G, depth + 1, max_depth))

    def _key(self):
        return self._depth

    def depth(self):
        return self._depth

    def active_players(self):
        return [0, 1]

    def num_actions(self, player_id):
        return self._G.shape[player_id]

    def payoffs(self, actions):
        payoff = self._G[actions[0], actions[1]]
        return payoff, 1. - payoff

    def successors(self, actions):
        return self._successors

    def transitions(self, actions):
        return [1.]

    def descendants(self):
        return self._successors


class RepeatedGame(Game):

    def __init__(self, config):
        rows = config.get("rows", 10)
        columns = config.get("columns", rows)
        game = build_game(config.get("game", "zero_sum"),
                          rows,
                          columns,
                          config.get("config", {}))

        self._rounds = config.get("rounds", 1)
        self._initial_states = [RepeatedGameState(game.G, 1, self._rounds)]

    def num_players(self):
        return 2

    def max_payoff(self):
        return self._rounds

    def initial_states(self):
        return self._initial_states

    def initial_distribution(self):
        return [1.]
