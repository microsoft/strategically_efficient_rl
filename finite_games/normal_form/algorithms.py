import numpy as np

from normal_form.solvers import build_solver
from normal_form.sparring_exp3 import SparringExp3
from normal_form.sparring_rmax import SparringRMax
from normal_form.explore_exploit import ExploreExploit
from normal_form.sparring_ppo import SparringPPO
from normal_form.nash_v import NashV


class Expert:

    def __init__(self, N, M, T, config):
        self._game = np.zeros((N, M,))

        # Get solver
        solver = config.get("solver", "linear_programming")
        solver_config = config.get("solver_config", {})
        self._solver = build_solver(solver, solver_config)

    def sample(self, G):
        self._game = G.G.copy()

    def strategies(self):
        row_strategies, column_strategies, _ = self._solver(self._game, 1.0 - self._game)
        return row_strategies, column_strategies

    def __repr__(self):
        return "expert"


class UniformRandom:

    def __init__(self, N, M, T, config):
        self._row_strategy = np.ones(N, dtype=float) / N
        self._column_strategy = np.ones(M, dtype=float) / M

    def sample(self, G):
        pass

    def strategies(self):
        return self._row_strategy, self._column_strategy

    def __repr__(self):
        return "random"


ALGORITHMS = {
    "random": UniformRandom,
    "expert": Expert,
    "exp3": SparringExp3,
    "rmax": SparringRMax,  # TODO: Need to check implementation
    "explore_exploit": ExploreExploit,
    "ppo": SparringPPO,
    "nash_v": NashV,
}

def build_algorithm(alg, N, M, T, config={}):
    if alg not in ALGORITHMS:
        raise ValueError(f"Leanring algorithm '{alg}' is not defined")

    return ALGORITHMS[alg](N, M, T, config)
