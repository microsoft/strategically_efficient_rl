from collections import defaultdict
import numpy as np
import scipy.optimize


class NashQ:

    def __init__(self, env, config):
        self._env = env

        # Check that we are in a two-player game
        assert env.num_players() == 2, "Nash-Q is only implemented for 2-player games"
        
        # Get configuration
        self._iteration_episodes = config.get("iteration_episodes", 10)
        self._learning_rate = config.get("learning_rate", 1.0)
        self._initial_value = config.get("initial_value", env.max_payoff())
        self._solver = config.get("solver", "revised simplex")
        self._epsilon = config.get("epsilon", 0.0)

        # Initialize upper and lower value functions
        self._Q = dict()
        self._V = defaultdict(lambda: self._initial_value)

        # Initialize strategies
        self._strategies = dict()

    def _solve_row(self, G):
        N, M = G.shape

        # Objective - maximize game value 'v'
        c = np.zeros(1 + N, dtype=np.float64)
        c[0] = -1.0

        # Find a row-strategy that receives payoff at least 'v' for all column actions
        A_ub = np.concatenate((np.ones((M,1,), dtype=np.float64), -G.T,), axis=1)
        b_ub = np.zeros(M, dtype=np.float64)

        # Ensure that row strategy is a distribution
        A_eq = np.ones((1, 1 + N,), dtype=np.float64)
        A_eq[0, 0] = 0.0
        b_eq = np.ones(1, dtype=np.float64)

        bounds = [(0.0,None,)] * (1 + N)
        bounds[0] = (None, None)

        # Use SciPy to solve the game
        result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, self._solver)

        # Normalize strategy
        strategy = np.clip(result.x[1:], 0.0, 1.0)
        strategy /= np.sum(strategy)

        if np.isnan(strategy).any():
            print(result)
            exit(1)

        return strategy

    def _solve_nash(self, G):
        row_strategy = self._solve_row(G)
        column_strategy = self._solve_row(-G.T)

        return row_strategy, column_strategy

    def _update_simultaneous(self, state, actions, rewards, next_state):

        # Get number of actions for each player
        row_actions = state.num_actions(0)
        column_actions = state.num_actions(1)

        # Compute learning rate - constant for now
        alpha = self._learning_rate

        # Initialize Q-function for the current state if necessary
        if state not in self._Q:
            self._Q[state] = np.full((row_actions, column_actions,), self._initial_value, dtype=np.float64)
        
        # Compute Q-target
        q_target = rewards[0]

        if next_state is not None:
            q_target += self._V[next_state]

        # Update Q-function
        self._Q[state][actions[0], actions[1]] = (1.0 - alpha) * self._Q[state][actions[0], actions[1]] + alpha * q_target

        # Recompute strategies
        row_strategy, column_strategy = self._solve_nash(self._Q[state])
        self._strategies[state] = (row_strategy, column_strategy)

        # Recompute value function
        self._V[state] = row_strategy.dot(self._Q[state]).dot(column_strategy)

    def _update_turn_based(self, state, actions, rewards, next_state):

        # Get current player ID
        player_id = state.active_players()[0]

        # Get number of actions for current player
        num_actions = state.num_actions(player_id)

        # Compute learning rate - constant for now
        alpha = self._learning_rate

        # Initialize Q-bounds the current state if necessary
        if state not in self._Q:
            self._Q[state] = np.full(num_actions, self._initial_value, dtype=np.float64)
        
        # Compute Q-target
        q_target = rewards[0]

        if next_state is not None:
            q_target += self._V[next_state]
        
        # Update Q-function
        self._Q[state][actions[0]] = (1.0 - alpha) * self._Q[state][actions[0]] + alpha * q_target

        # Recompute strategy
        q_function = self._Q[state]

        if 0 == player_id:
            q_target = np.max(q_function)
        else:
            q_target = np.min(q_function)

        strategy = np.zeros(num_actions, dtype=np.float64)

        for action in range(num_actions):
            if q_function[action] == q_target:
                strategy[action] = 1.0

        strategy = strategy / np.sum(strategy)
        self._strategies[state] = strategy

        # Recompute value function
        self._V[state] = np.sum(q_function * strategy)

    def _actions(self, state):
        if len(state.active_players()) == 1:
            num_actions = state.num_actions(state.active_players()[0])

            if state not in self._strategies or np.random.random() < self._epsilon:
                strategy = np.ones(num_actions, dtype=np.float64) / num_actions
            else:
                strategy = self._strategies[state]

            return [np.random.choice(num_actions, p=strategy)]
        else:
            row_actions = state.num_actions(0)
            column_actions = state.num_actions(1)

            if state not in self._strategies or np.random.random() < self._epsilon:
                row_strategy = np.ones(row_actions, dtype=np.float64) / row_actions
                column_strategy = np.ones(column_actions, dtype=np.float64) / column_actions
            else:
                row_strategy, column_strategy = self._strategies[state]

            row_action = np.random.choice(row_actions, p=row_strategy)
            column_action = np.random.choice(column_actions, p=column_strategy)

            return [row_action, column_action]
    
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
                current_action = self._actions(current_state)
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
                
                if len(state.active_players()) == 1:
                    self._update_turn_based(state, actions, rewards, next_state)
                else:
                    self._update_simultaneous(state, actions, rewards, next_state)

        # Return number of steps and episodes sampled, and episode statistics
        return total_steps, self._iteration_episodes, self._env.pull_stats()

    def strategy(self, state, player_id):
        if state not in self._strategies:
            num_actions = state.num_actions(player_id)
            return np.ones(num_actions, dtype=np.float64) / num_actions
        elif len(state.active_players()) == 1:
            return self._strategies[state]
        else:
            return self._strategies[state][player_id]
