import numpy as np
from normal_form.solvers import build_solver


class SparringRMax:

    def __init__(self, N, M, T, config):
        delta = config.get("delta", 0.05)
        epsilon = config.get("epsilon", 0.05)

        self._required_samples = np.ceil(-np.log(delta / (2 * N * M)) / (2 * epsilon))
        
        self._counts = np.zeros((N,M,))
        self._row_totals = np.zeros((N,M,))
        self._column_totals = np.zeros((N,M,))
        
        self._row_payoffs = np.ones((N,M,))
        self._column_payoffs = np.ones((M,N,))

        self._row_strategy = np.ones(N) / N
        self._column_strategy = np.ones(M) / M

        # Get solver
        solver = config.get("solver", "hedge")
        solver_config = config.get("solver_config", {})
        self._solver = build_solver(solver, solver_config)

    def sample(self, G):

        # Sample payoff
        row_action = np.random.choice(len(self._row_strategy), p=self._row_strategy)
        column_action = np.random.choice(len(self._column_strategy), p=self._column_strategy)
        row_payoff, column_payoff = G.sample(row_action, column_action)
        
        if self._counts[row_action, column_action] < self._required_samples:
            self._counts[row_action, column_action] += 1.0
            self._row_totals[row_action, column_action] += row_payoff
            self._column_totals[row_action, column_action] += column_payoff

            if self._counts[row_action, column_action] >= self._required_samples:
                row_mean = self._row_totals[row_action, column_action] / self._counts[row_action, column_action]
                self._row_payoffs[row_action, column_action] = row_mean

                column_mean = self._column_totals[row_action, column_action] / self._counts[row_action, column_action]
                self._column_payoffs[column_action, row_action] = 1.0 - column_mean

                self._row_strategy, self._column_strategy, _ = self._solver(self._row_payoffs, self._column_payoffs)

    def strategies(self):
        return self._row_strategy, self._column_strategy

    def __repr__(self):
        return f"sparring_rmax_{self._required_samples}_{self._solver}"
