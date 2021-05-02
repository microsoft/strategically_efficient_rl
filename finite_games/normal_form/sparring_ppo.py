import numpy as np

DEFAULT_CONFIG = {
    "regularization": "fixed",
    "kl_scale": 0.01,
    "initial_kl_coeff": 0.2,
    "kl_target": 0.05,
    "batch_size": 10,
    "num_grad_iter": 100,
    "learning_rate": 0.00001,
    "grad_clip": 1.0,
    "implcit_exploration": 0.001,
    "averaging": False,
}


class PPOLearner:

    def __init__(self, num_actions, horizon, config):
        self._config = config
        self._num_actions = num_actions
        
        # Initialize parameters and strategies
        self._logits = np.zeros(num_actions)
        self._strategy = np.ones(num_actions) / num_actions

        self._action_counts = np.ones(num_actions)

        # Data buffers
        self._actions = []
        self._payoffs = []

        # KL coefficient
        regularization = config["regularization"]

        if "fixed" == regularization: # Fixed weight - derived from mirror descent bounds
            horizon = config.get("horizon", horizon)
            self._kl_coeff = config["kl_scale"] * np.sqrt(2 * np.log(num_actions) / horizon)
        elif "adaptive" == regularization:  # Adaptive weight - original PPO paper
            self._kl_coeff = config["initial_kl_coeff"]
        else:
            raise ValueError(f"PPO Regularization method '{regularization}' is not defined")

    def _gradient(self, new_strategy):
        exploration = self._config["implcit_exploration"]

        # Compute advantage gradient
        gradient = np.zeros(self._num_actions)

        for action, payoff in zip(self._actions, self._payoffs):
            scale = payoff * new_strategy[action] / (self._strategy[action] + exploration)
            gradient += scale * new_strategy
            gradient[action] -= scale

        # Compute KL gradient
        gradient += self._kl_coeff * (new_strategy - self._strategy)

        # Clip gradient if needed
        if self._config["grad_clip"] is not None:
            norm = np.sqrt(np.sum(np.square(gradient)))

            if norm > self._config["grad_clip"]:
                gradient *= self._config["grad_clip"] / norm

        return gradient

    def _update(self):

        # Do gradient updates
        new_logits = self._logits.copy()
        new_strategy = self._strategy.copy()

        for _ in range(self._config["num_grad_iter"]):
            new_logits -= self._config["learning_rate"] * self._gradient(new_strategy)
            new_logits -= np.mean(new_logits)
            
            weights = np.exp(new_logits - np.max(new_logits))
            new_strategy = weights / np.sum(weights)

        # Update KL coefficients
        if self._config["regularization"]:
            kl = np.sum(self._strategy * (np.log(self._strategy) - np.log(new_strategy)))

            if kl < self._config["kl_target"] / 1.5:
                self._kl_coeff /= 2.0
            elif kl > self._config["kl_target"] * 1.5:
                self._kl_coeff *= 2.0

        self._logits = new_logits
        self._strategy = new_strategy

        self._actions = []
        self._payoffs = []

    def observe(self, action, payoff):
        self._actions.append(action)
        self._payoffs.append(payoff)

        self._action_counts[action] += 1.0

        if len(self._actions) >= self._config["batch_size"]:
            self._update()

    def sample(self):
        return np.random.choice(self._num_actions, p=self._strategy)

    @property
    def strategy(self):
        if self._config["averaging"]:
            return self._action_counts / np.sum(self._action_counts)
        else:
            return self._strategy


class SparringPPO:

    def __init__(self, N, M, T, config):
        self._config = DEFAULT_CONFIG.copy()
        self._config.update(config)

        self._row_learner = PPOLearner(N, T, self._config)
        self._column_learner = PPOLearner(M, T, self._config)

    def sample(self, G):
        row_action = self._row_learner.sample()
        column_action = self._column_learner.sample()

        row_payoff, column_payoff = G.sample(row_action, column_action)

        self._row_learner.observe(row_action, row_payoff)
        self._column_learner.observe(column_action, column_payoff)

    def strategies(self):
        return self._row_learner.strategy, self._column_learner.strategy

    def __repr__(self):
        return "sparring_ppo"
