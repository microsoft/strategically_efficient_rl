from extensive_form.games.repeated_game import RepeatedGame
from extensive_form.games.deep_sea import DeepSea
from extensive_form.games.decoy_deep_sea import DecoyDeepSea
from extensive_form.games.double_decoy_deep_sea import DoubleDecoyDeepSea
from extensive_form.games.alpha_beta import AlphaBeta
from extensive_form.games.grid_world import GridWorld
from extensive_form.games.decoy_tree import DecoyTree
from extensive_form.games.tree_game import TreeGame
from extensive_form.games.simultaneous_tree_game import SimultaneousTreeGame
from extensive_form.games.stochastic_game import StochasticGame

GAMES = {
    "repeated": RepeatedGame,
    "deep_sea": DeepSea,
    "decoy_deep_sea": DecoyDeepSea,
    "double_decoy_deep_sea": DoubleDecoyDeepSea,
    "grid_world": GridWorld,
    "alpha_beta": AlphaBeta,
    "decoy_tree": DecoyTree,
    "tree_game": TreeGame,
    "simultaneous_tree_game": SimultaneousTreeGame,
    "stochastic_game": StochasticGame,
}


def build_game(game, config={}):
    if game not in GAMES:
        raise ValueError(f"Game type '{game}' is not defined")

    return GAMES[game](config)
