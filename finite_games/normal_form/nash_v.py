import numpy as np


class NashV:

    def __init__(self, N, M, T, config):
        self._N = N
        self._M = M

        self._alpha = config.get("alpha", 1.0)
        self._gamma = config.get("gamma", 1.0)

        # Sample counter
        self._sample_count = 0

        # Reward estimates
        self._row_logits= np.zeros(N)
        self._column_logits = np.zeros(M)

        # Strategies
        self._row_strategy = np.ones(N) / N
        self._column_strategy = np.ones(M) / M

        self._row_average_strategy = np.ones(N) / N
        self._column_average_strategy = np.ones(M) / M

    def sample(self, G): # Generate samples from the game G - return the number of samples

        # Sample payoff
        row_action = np.random.choice(self._N, p=self._row_strategy)
        column_action = np.random.choice(self._M, p=self._column_strategy)
        row_payoff, column_payoff = G.sample(row_action, column_action)

        # Update logits
        self._sample_count += 1
        
        alpha = self._alpha * 2 / (1 + self._sample_count)
        row_gamma = self._gamma * np.sqrt(np.log(self._N) / (self._N * self._sample_count))
        column_gamma = self._gamma * np.sqrt(np.log(self._M) / (self._M * self._sample_count))

        row_losses = np.zeros(self._N)
        row_losses[row_action] = (1 - row_payoff) / (self._row_strategy[row_action] + row_gamma)

        column_losses = np.zeros(self._M)
        column_losses[column_action] = (1 - column_payoff) / (self._column_strategy[column_action] + column_gamma)

        self._row_logits = (1 - alpha) * self._row_logits + alpha * row_losses
        self._column_logits = (1 - alpha) * self._column_logits + alpha * column_losses

        # Update strategies
        row_advantages = self._row_logits - np.max(self._row_logits)
        row_weights = np.exp(-(row_gamma / alpha) * row_advantages)

        column_advantages = self._column_logits - np.max(self._column_logits)
        column_weights = np.exp(-(column_gamma / alpha) * column_advantages)

        self._row_strategy = row_weights / np.sum(row_weights)
        self._column_strategy = column_weights / np.sum(column_weights)

        assert all(np.isfinite(self._row_logits)), "One or more row logits became infinite"
        assert all(np.isfinite(self._column_logits)), "One or more column logits became infinite"

        # Update average strategies
        self._row_average_strategy = (1 - alpha) * self._row_average_strategy + alpha * self._row_strategy
        self._column_average_strategy = (1 - alpha) * self._column_average_strategy + alpha * self._column_strategy

    def strategies(self): # Return the test strategies, may differ from the sampling strategies
        return self._row_average_strategy, self._column_average_strategy
    
    def __repr__(self):
        return f"nash_v_alpha_{self._alpha}_gamma_{self._gamma}"
