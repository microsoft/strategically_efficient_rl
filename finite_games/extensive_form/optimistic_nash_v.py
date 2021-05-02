from collections import defaultdict
import numpy as np


class NashVInstance:

    def __init__(self, player_id, max_payoff, eta, gamma, beta, horizon, initial_value, averaging):
        self._player_id = player_id
        self._max_payoff = max_payoff
        self._eta = eta
        self._gamma = gamma
        self._beta = beta
        self._horizon = horizon
        self._averaging = averaging

        self._count = defaultdict(lambda: 0)
        self._V = defaultdict(lambda: initial_value)

        self._L = dict()
        self._explore_strategies = dict()
        self._exploit_strategies = dict()

    def update(self, state, action, reward, next_state):
        num_actions = state.num_actions(self._player_id)

        # Update visitation counts
        count = self._count[state] + 1
        self._count[state] = count

        # Compute learning rates and exploration bonuses
        alpha = (self._horizon + 1) / (self._horizon + count)
        eta = self._eta * np.sqrt(np.log(num_actions) / (num_actions * count))
        gamma = self._gamma * np.sqrt(np.log(num_actions) / (num_actions * count))
        beta = self._beta / np.sqrt(count)

        # Compute value target
        value_pred = reward

        if next_state is not None:
            value_pred += self._V[next_state]

        # Update value function
        self._V[state] = min(self._max_payoff, (1 - alpha) * self._V[state] + alpha * (value_pred + beta))

        # Update action losses if the agent took an action
        if action is not None:

            # Get the number of actions for the current state
            num_actions = state.num_actions(self._player_id)

            # Initialize loss estimator and strategy history
            if state not in self._L:
                self._L[state] = np.zeros(num_actions)  # Should this be initialized to zero?
                self._explore_strategies[state] = np.ones(num_actions) / num_actions
                self._exploit_strategies[state] = np.ones(num_actions) / num_actions

            # Update loss estimator
            loss = self._max_payoff - value_pred
            weight = self._explore_strategies[state][action] + gamma

            ell = np.zeros(num_actions)
            ell[action] = loss / weight

            losses = (1 - alpha) * self._L[state] + alpha * ell
            self._L[state] = losses

            # Compute new exploration and exploitation strategies
            logits = losses - np.max(losses)
            weights = np.exp(-(eta / alpha) * logits)

            strategy = weights / np.sum(weights)
            self._explore_strategies[state] = strategy

            if self._averaging:
                self._exploit_strategies[state] = (1 - alpha) * self._exploit_strategies[state] + alpha * strategy
            else:
                self._exploit_strategies[state] = strategy

    def explore(self, state):
        num_actions = state.num_actions(self._player_id)

        if state not in self._explore_strategies:
            return np.random.randint(state.num_actions(self._player_id))
        
        return np.random.choice(num_actions, p=self._explore_strategies[state])

    def strategy(self, state):
        num_actions = state.num_actions(self._player_id)

        if state not in self._exploit_strategies:
            return np.ones(num_actions) / num_actions
        
        return self._exploit_strategies[state]


class OptimisticNashV:

    def __init__(self, env, config):
        self._env = env

        # Get the number of episodes to sample at each iteration
        self._iteration_episodes = config.get("iteration_episodes", 10)

        # Initialize Nash-V learners for each player
        self._learners = []
        
        for player_id in range(env.num_players()):
            self._learners.append(NashVInstance(player_id, 
                                                config.get("max_payoff", env.max_payoff()), 
                                                config.get("eta", 1.0),
                                                config.get("gamma", 1.0),
                                                config.get("beta", 1.0),
                                                config.get("horizon", env.max_depth()),
                                                config.get("initial_value", env.max_payoff()),
                                                config.get("averaging", False)))

    def train(self):

        # Step counter
        total_steps = 0

        for _ in range(self._iteration_episodes):

            # Reset environment
            current_state = self._env.reset()

            # Initialize history
            state_history = []
            action_history = []
            reward_history = []

            while current_state is not None:

                # Append current state to state history
                state_history.append(current_state)

                # Get actions for all active players
                current_action = []

                for player_id in current_state.active_players():
                    current_action.append(self._learners[player_id].explore(current_state))

                action_history.append(current_action)

                # Take joint action in environment
                current_state, rewards = self._env.step(current_action)

                # Append payoffs to payoff history
                reward_history.append(rewards)

            # Update step count
            total_steps += len(state_history)

            # Do updates for each player - work forward through history
            for step in range(len(state_history)):
                state = state_history[step]
                actions = action_history[step]
                rewards = reward_history[step]

                if step + 1 < len(state_history):
                    next_state = state_history[step + 1]
                else:
                    next_state = None

                for player_id, learner in enumerate(self._learners):
                    reward = rewards[player_id]

                    if player_id in state.active_players():
                        action = actions[state.active_players().index(player_id)]
                    else:
                        action = None
                    
                    learner.update(state, action, reward, next_state)

        # Return number of steps and episodes sampled, and episode statistics
        return total_steps, self._iteration_episodes, self._env.pull_stats()

    def strategy(self, state, player_id):
        return self._learners[player_id].strategy(state)
