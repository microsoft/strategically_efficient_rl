import numpy as np
import scipy.optimize


class LinearProgramming:

    def __init__(self, config):
        self._method = config.get("method", "revised simplex")

    def _solve_row(self, G):
        M, N = G.shape

        c = np.zeros(M + 1)
        c[0] = 1.0

        A_ub = np.concatenate((-np.ones((N,1,)), -G.T,), axis=1)
        b_ub = np.zeros(N)

        A_eq = np.ones((1, M + 1,))
        A_eq[0, 0] = 0.0
        b_eq = np.ones(1)

        bounds = [(0.0,None,)] * (M + 1)
        bounds[0] = (None, None)

        # Use SciPy to solve the game
        result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, self._method)
        
        # Return row strategy and row-player value
        return result.x[1:], -result.x[0]

    def _normalize(self, strategy):
        strategy = np.clip(strategy, 0, 1)
        return strategy / np.sum(strategy)

    def __call__(self, row_payoffs, column_payoffs=None):
        if column_payoffs is None:
            column_payoffs = -row_payoffs
        
        row_strategy, row_value = self._solve_row(row_payoffs)
        column_strategy, column_value = self._solve_row(column_payoffs.T)

        row_strategy = self._normalize(row_strategy)
        column_strategy = self._normalize(column_strategy)

        info = {
            "row_value": row_value,
            "column_value": column_value,
        }

        return row_strategy, column_strategy, info
        
    def __repr__(self):
        return "lp"