from collections import defaultdict
import numpy as np


class QLearner:

    def __init__(self, player_id, max_depth, max_payoff, config):
        self._player_id = player_id

        self._epsilon = config.get("epsilon", 0.01)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 1.)
        self._horizon = config.get("horizon", max_depth)
        self._initial = config.get("initial", max_payoff)
        self._exploit = config.get("exploit", False)
        self._averaging = config.get("averaging", False)

        self._Q = dict()
        self._L = dict()

        self._counts = defaultdict(lambda: 0.)

        self._explore_strategies = dict()
        self._exploit_strategies = dict()

    def _strategy(self, num_actions, q_function):
        q_target = np.max(q_function)
        strategy = np.zeros(num_actions)
        
        for action in range(num_actions):
            if q_function[action] == q_target:
                strategy[action] = 1.

        return strategy / np.sum(strategy)

    def update(self, state, action, reward, next_state):
        num_actions = max(1, state.num_actions(self._player_id))

        # Initialize Q-value if necessary
        if state not in self._Q:
            self._Q[state] = np.full(num_actions, self._initial)
            self._L[state] = np.full(num_actions, 0.0)

        # Update visitation count
        self._counts[state] += 1.
        count = self._counts[state]

        # Compute value target
        value_pred = reward + self._beta / np.sqrt(count)
        extrinsic_value_pred = reward

        if next_state is not None:

            # Initialize next Q-value if necessary
            if next_state not in self._Q:
                self._Q[next_state] = np.full(next_state.num_actions(self._player_id), self._initial)
                self._L[next_state] = np.full(next_state.num_actions(self._player_id), 0.0)

            value_pred += self._gamma * np.max(self._Q[next_state])
            extrinsic_value_pred += self._gamma * np.max(self._L[next_state])

        # Compute learning rate
        alpha = (self._horizon + 1.) / (self._horizon + count)

        # Update Q-value
        self._Q[state][action] = (1. - alpha) * self._Q[state][action] + alpha * value_pred
        self._L[state][action] = (1. - alpha) * self._L[state][action] + alpha * extrinsic_value_pred

        # Update exploration and exploitation strategies
        strategy = self._strategy(num_actions, self._Q[state])
        self._explore_strategies[state] = strategy

        if self._exploit:
            strategy = self._strategy(num_actions, self._L[state])
        
        if self._averaging:
            if state not in self._exploit_strategies:
                self._exploit_strategies[state] = np.ones(num_actions) / num_actions

            self._exploit_strategies[state] = (1 - alpha) * self._exploit_strategies[state] + alpha * strategy
        else:
            self._exploit_strategies[state] = strategy

    def explore(self, state):
        num_actions = max(1, state.num_actions(self._player_id))

        if state not in self._explore_strategies or np.random.random() <= self._epsilon:
            return np.random.choice(num_actions)
        
        return np.random.choice(num_actions, p=self._explore_strategies[state])

    def strategy(self, state):
        num_actions = max(1, state.num_actions(self._player_id))

        if state not in self._exploit_strategies:
            return np.ones(num_actions) / num_actions
        
        return self._exploit_strategies[state]


class OptimisticQLearning:

    def __init__(self, env, config):
        self._env = env

        # Get the number of episodes to sample at each iteration
        self._iteration_episodes = config.get("iteration_episodes", 10)

        # Initialize Nash-V learners for each player
        self._learners = []
        
        for player_id in range(env.num_players()):
            self._learners.append(QLearner(player_id, env.max_depth(), env.max_payoff(), config))

    def train(self):

        # Step counter
        total_steps = 0

        # Update diagnostics
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
                        action = 0
                    
                    learner.update(state, action, reward, next_state)

        # Return number of steps and episodes sampled, and episode statistics
        return total_steps, self._iteration_episodes, self._env.pull_stats()

    def strategy(self, state, player_id):
        return self._learners[player_id].strategy(state)
