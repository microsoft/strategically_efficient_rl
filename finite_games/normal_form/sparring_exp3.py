import numpy as np


class SparringExp3:

    def __init__(self, N, M, T, config):
        self._N = N
        self._M = M

        # Override externally provided horizon if given
        T = config.get("horizon", T)

        # Update scales
        self._row_beta = np.sqrt(2 * np.log(N + 1) / (T * N))
        self._column_beta = np.sqrt(2 * np.log(M + 1) / (T * M))

        if config.get("implicit_exploration", True):
            self._row_gamma = self._row_beta / 2
            self._column_gamma = self._column_beta / 2
        else:
            self._row_gamma = 0
            self._column_gamma = 0

        # Override theoretical values with external constants if requested
        if "beta" in config:
            self._row_beta = config["beta"]
            self._column_beta = config["beta"]
        
        if "gamma" in config:
            self._row_gamma = config["gamma"]
            self._column_gamma = config["gamma"]

        # Reward estimates
        self._row_payoffs= np.zeros(N)
        self._column_payoffs = np.zeros(M)

        # Action counts - for empirical strategies
        self._row_counts = np.zeros(N)
        self._column_counts = np.zeros(M)

    def sample(self, G): # Generate samples from the game G - return the number of samples

        # Compute strategies
        row_advantages = self._row_payoffs - np.max(self._row_payoffs)
        row_weights = np.exp(self._row_beta * row_advantages)

        column_advantages = self._column_payoffs - np.max(self._column_payoffs)
        column_weights = np.exp(self._column_beta * column_advantages)

        row_strategy = row_weights / np.sum(row_weights)
        column_strategy = column_weights / np.sum(column_weights)

        # Sample payoff
        row_action = np.random.choice(self._N, p=row_strategy)
        column_action = np.random.choice(self._M, p=column_strategy)
        row_payoff, column_payoff = G.sample(row_action, column_action)

        # Update weights
        self._row_payoffs[row_action] -= (1.0 - row_payoff) / (row_strategy[row_action] + self._row_gamma)
        self._row_payoffs += 1.0

        self._column_payoffs[column_action] -= (1.0 - column_payoff) / (column_strategy[column_action] + self._column_gamma)
        self._column_payoffs += 1.0

        assert all(np.isfinite(self._row_payoffs)), "One or more row payoffs became infinite"
        assert all(np.isfinite(self._column_payoffs)), "One or more column payoffs became infinite"

        # Update counts
        self._row_counts[row_action] += 1.0
        self._column_counts[column_action] += 1.0

    def strategies(self): # Return the test strategies, may differ from the sampling strategies
        row_strategy = self._row_counts / np.sum(self._row_counts)
        column_strategy = self._column_counts / np.sum(self._column_counts)

        return row_strategy, column_strategy
    
    def __repr__(self):
        return f"sparring_exp3_beta_{self._row_beta}_{self._column_beta}_gamma_{self._row_gamma}_{self._column_gamma}"
