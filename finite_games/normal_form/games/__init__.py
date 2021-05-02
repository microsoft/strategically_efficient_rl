from normal_form.games.zero_sum import RandomZeroSumGame
from normal_form.games.bernoulli import RandomBernoulliGame
from normal_form.games.roshambo import Roshambo
from normal_form.games.hybrid_roshambo import HybridRoshambo
from normal_form.games.unique_equilibrium import UniqueEquilibrium

GAMES = {
    "zero_sum": RandomZeroSumGame,
    "bernoulli": RandomBernoulliGame,
    "roshambo": Roshambo,
    "hybrid_roshambo": HybridRoshambo,
    "unique": UniqueEquilibrium, 
}


def build_game(game, N, M, config={}):
    if game not in GAMES:
        raise ValueError(f"Game type '{game}' is not defined")

    return GAMES[game](N, M, config)
