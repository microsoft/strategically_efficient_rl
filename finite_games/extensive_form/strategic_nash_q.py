from collections import defaultdict
import numpy as np
import scipy.optimize


class StrategicNashQ:

    def __init__(self, env, config):
        self._env = env

        # Check that we are in a two-player game
        assert env.num_players() == 2, "Strategic Nash-Q is only implemented for 2-player games"
        
        # Get configuration
        self._iteration_episodes = config.get("iteration_episodes", 10)
        self._horizon = config.get("horizon", env.max_depth())
        self._beta = config.get("beta", 0.0)
        self._solver = config.get("solver", "revised simplex")
        self._averaging = config.get("averaging", False)

        # Set initial value - allows us to override optimistic initialization
        self._initial_value = config.get("initial_value", env.max_payoff())

        # Initialize upper and lower value functions
        self._Q_upper = dict()
        self._Q_lower = dict()

        self._V_upper = defaultdict(lambda: self._initial_value)
        self._V_lower = defaultdict(lambda: 0.0)

        # Initialize counts
        self._counts = dict()

        # Initialize strategies
        self._explore_strategies = dict()
        self._exploit_strategies = dict()

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
    
    def _solve_turn_based(self, player_id, num_actions, q_function):
        if 0 == player_id:
            q_target = np.max(q_function)
        else:
            q_target = np.min(q_function)

        strategy = np.zeros(num_actions, dtype=np.float64)

        for action in range(num_actions):
            if q_function[action] == q_target:
                strategy[action] = 1.0

        return strategy / np.sum(strategy)

    def _update_simultaneous(self, state, actions, rewards, next_state):

        # Get number of actions for each player
        row_actions = state.num_actions(0)
        column_actions = state.num_actions(1)

        # Initialize counts if necessary
        if state not in self._counts:
            self._counts[state] = np.zeros((row_actions, column_actions), dtype=np.int64)

        # Get visitation count
        self._counts[state][actions[0], actions[1]] += 1
        count = self._counts[state][actions[0], actions[1]]

        # Compute learning rates and exploration bonuses
        alpha = (self._horizon + 1) / (self._horizon + count)
        beta = self._beta / np.sqrt(count)

        # Initialize Q-bounds the current state if necessary
        if state not in self._Q_upper:
            self._Q_upper[state] = np.full((row_actions, column_actions), self._initial_value, dtype=np.float64)
            self._Q_lower[state] = np.full((row_actions, column_actions), 0.0, dtype=np.float64)
        
        # Compute Q-bound updates - use the max-player reward
        q_upper = rewards[0] + beta
        q_lower = rewards[0] - beta

        if next_state is not None:
            q_upper += self._V_upper[next_state]
            q_lower += self._V_lower[next_state]
        
        q_upper = min(self._initial_value, q_upper)
        q_lower = max(0.0, q_lower)

        # Update Q-bounds for current state and action
        self._Q_upper[state][actions[0], actions[1]] = (1.0 - alpha) * self._Q_upper[state][actions[0], actions[1]] + alpha * q_upper
        self._Q_lower[state][actions[0], actions[1]] = (1.0 - alpha) * self._Q_lower[state][actions[0], actions[1]] + alpha * q_lower

        # Recompute strategies
        row_explore, column_exploit = self._solve_nash(self._Q_upper[state])
        row_exploit, column_explore = self._solve_nash(self._Q_lower[state])

        self._explore_strategies[state] = {
            0: row_explore,
            1: column_explore
        }

        if state not in self._exploit_strategies:
            self._exploit_strategies[state] = dict()

            if self._averaging:
                self._exploit_strategies[state][0] = np.ones(row_actions, dtype=np.float64) / row_actions
                self._exploit_strategies[state][1] = np.ones(column_actions, dtype=np.float64) / column_actions

        if self._averaging:
            self._exploit_strategies[state][0] = (1 - alpha) * self._exploit_strategies[state][0] + alpha * row_exploit
            self._exploit_strategies[state][1] = (1 - alpha) * self._exploit_strategies[state][1] + alpha * column_exploit
        else:
            self._exploit_strategies[state][0] = row_exploit
            self._exploit_strategies[state][1] = column_exploit

        # Recompute V-bounds
        self._V_upper[state] = row_explore.dot(self._Q_upper[state]).dot(column_exploit)
        self._V_lower[state] = row_exploit.dot(self._Q_lower[state]).dot(column_explore)
        
    def _update_turn_based(self, state, actions, rewards, next_state):

        # Get current player ID
        player_id = state.active_players()[0]

        # Get number of actions for current player
        num_actions = state.num_actions(player_id)

        # Initialize counts if necessary
        if state not in self._counts:
            self._counts[state] = np.zeros(num_actions, dtype=np.int64)

        # Get visitation count
        self._counts[state][actions[0]] += 1
        count = self._counts[state][actions[0]]

        # Compute learning rates and exploration bonuses
        alpha = (self._horizon + 1) / (self._horizon + count)
        beta = self._beta / np.sqrt(count)

        # Initialize Q-bounds the current state if necessary
        if state not in self._Q_upper:
            self._Q_upper[state] = np.full(num_actions, self._initial_value, dtype=np.float64)
            self._Q_lower[state] = np.full(num_actions, 0.0, dtype=np.float64)
        
        # Compute Q-bound updates - use the max-player reward
        q_upper = rewards[0] + beta
        q_lower = rewards[0] - beta

        if next_state is not None:
            q_upper += self._V_upper[next_state]
            q_lower += self._V_lower[next_state]
        
        q_upper = min(self._initial_value, q_upper)
        q_lower = max(0.0, q_lower)

        # Update Q-bounds for current state and action
        self._Q_upper[state][actions[0]] = (1.0 - alpha) * self._Q_upper[state][actions[0]] + alpha * q_upper
        self._Q_lower[state][actions[0]] = (1.0 - alpha) * self._Q_lower[state][actions[0]] + alpha * q_lower

        # Recompute strategies - how does this work?
        if 0 == player_id:
            explore_strategy = self._solve_turn_based(player_id, num_actions, self._Q_upper[state])
            exploit_strategy = self._solve_turn_based(player_id, num_actions, self._Q_lower[state])
        else:
            explore_strategy = self._solve_turn_based(player_id, num_actions, self._Q_lower[state])
            exploit_strategy = self._solve_turn_based(player_id, num_actions, self._Q_upper[state])
        
        self._explore_strategies[state] = explore_strategy

        # Update exploitation strategies
        if state not in self._exploit_strategies:
            self._exploit_strategies[state] = {
                player_id: np.ones(num_actions, dtype=np.float64) / num_actions
            }

        if self._averaging:
            self._exploit_strategies[state][player_id] = (1 - alpha) * self._exploit_strategies[state][player_id] + alpha * exploit_strategy
        else:
            self._exploit_strategies[state][player_id] = exploit_strategy

        # Recompute V-bounds
        if 0 == player_id:
            self._V_upper[state] = np.max(self._Q_upper[state])
            self._V_lower[state] = np.max(self._Q_lower[state])
        else:
            self._V_upper[state] = np.min(self._Q_upper[state])
            self._V_lower[state] = np.min(self._Q_lower[state])

    def _actions(self, state):
        if len(state.active_players()) == 1:
            num_actions = state.num_actions(state.active_players()[0])

            if state not in self._explore_strategies:
                strategy = np.ones(num_actions, dtype=np.float64) / num_actions
            else:
                strategy = self._explore_strategies[state]

            return [np.random.choice(num_actions, p=strategy)]
        else:
            row_actions = state.num_actions(0)
            column_actions = state.num_actions(1)

            if state not in self._explore_strategies:
                row_strategy = np.ones((row_actions,), dtype=np.float64) / row_actions
                column_strategy = np.ones((column_actions,), dtype=np.float64) / column_actions
            else:
                row_strategy = self._explore_strategies[state][0]
                column_strategy = self._explore_strategies[state][1]

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
        if state not in self._exploit_strategies:
            num_actions = state.num_actions(player_id)
            return np.ones(num_actions, dtype=np.float64) / num_actions
        else:
            return self._exploit_strategies[state][player_id]
