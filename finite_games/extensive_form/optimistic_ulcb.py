from collections import defaultdict
import numpy as np
import scipy.optimize


class OptimisticULCB:

    def __init__(self, env, config):
        self._env = env

        # Check that we are in a two-player game
        assert env.num_players() == 2, "Optimistic ULCB is only implemented for 2-player games"

        # Get configuration
        self._iteration_episodes = config.get("iteration_episodes", 10)
        self._horizon = config.get("horizon", env.max_depth())
        self._beta = config.get("beta", 0.0)
        self._solver = config.get("solver", "revised simplex")
        self._exploit = config.get("exploit", False)

        # Set initial value - allows us to override optimistic initialization
        self._initial_value = config.get("initial_value", env.max_payoff())

        # initialize game model
        self._counts = dict()
        self._transitions = dict()
        self._rewards = dict()

        # Initialize strategies
        self._explore_strategies = dict()
        self._exploit_strategies = dict()

    def _solve_CCE(self, A, B):
        N, M = A.shape

        c = np.zeros(N * M, dtype=np.float64)
        bounds = [(0.0, None)] * (N * M)

        b_eq = np.ones(1, dtype=np.float64)
        A_eq = np.ones((1,N * M,), dtype=np.float64)

        b_ub = np.zeros(N + M, dtype=np.float64)
        
        rows = []

        # Row bounds
        for a in range(N):
            A_diff = A[a] - A
            rows.append(A_diff.flatten())

        # Column bounds
        for b in range(M):
            B_diff = (B.T - B.T[b]).T
            rows.append(B_diff.flatten())

        A_ub = np.stack(rows)
        
        # Solve program
        result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, self._solver)
        
        # Normalize strategy
        strategy = np.clip(result.x.reshape((N, M,)), 0.0, 1.0)
        strategy /= np.sum(strategy)

        if np.isnan(strategy).any():
            print(result)
            exit(1)

        return strategy

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

    def _explore_strategy(self, state, Q_upper, Q_lower):
        player_id = state.active_players()[0]
        num_actions = state.num_actions(player_id)

        if 0 == player_id:
            q_function = Q_upper
            q_target = np.max(Q_upper)
        else:
            q_function = Q_lower
            q_target = np.min(Q_lower)

        strategy = np.zeros(num_actions, dtype=np.float64)

        for action in range(num_actions):
            if q_function[action] == q_target:
                strategy[action] = 1.0

        return strategy / np.sum(strategy)

    def _exploit_strategy(self, state, Q_upper, Q_lower):
        player_id = state.active_players()[0]
        num_actions = state.num_actions(player_id)

        if 0 == player_id:
            q_function = Q_lower
            q_target = np.max(Q_lower)
        else:
            q_function = Q_upper
            q_target = np.min(Q_upper)

        strategy = np.zeros(num_actions, dtype=np.float64)

        for action in range(num_actions):
            if q_function[action] == q_target:
                strategy[action] = 1.0

        return strategy / np.sum(strategy)

    def _recompute(self):

        # Initialize strategies
        self._explore_strategies = dict()
        self._exploit_strategies = dict()

        # Initialize value function bounds
        V_upper = defaultdict(lambda: self._initial_value)
        V_lower = defaultdict(lambda: 0) 
        
        # Sort visited states by depth
        state_space = sorted(self._counts.keys(), key=lambda state: -state.depth())

        # Solve game using value iteration, starting at the maximum depth
        for state in state_space:
            action_shape = self._counts[state].shape

            # Compute Q-function bounds
            Q_upper = np.zeros(action_shape, dtype=float)
            Q_lower = np.zeros(action_shape, dtype=float)

            for idx in np.ndindex(action_shape):
                if self._counts[state][idx] > 0:
                    
                    # Compute exploration bonus
                    beta = self._beta / np.sqrt(self._counts[state][idx])

                    # Compute average reward
                    reward = self._rewards[state][idx] / self._counts[state][idx]

                    # Get mean value of next state
                    next_upper = 0.
                    next_lower = 0.

                    if self._transitions[state][idx] is not None:
                        for next_state, count in self._transitions[state][idx].items():
                            next_upper += count * V_upper[next_state]
                            next_lower += count * V_lower[next_state]

                        next_upper /= self._counts[state][idx]
                        next_lower /= self._counts[state][idx]

                    # Set Q bounds
                    Q_upper[idx] = min(reward + next_upper + beta, self._horizon)
                    Q_lower[idx] = max(reward + next_lower - beta, 0.)

                else:
                    Q_upper[idx] = self._initial_value
                    Q_lower[idx] = 0.

            # Update strategies
            active_players = state.active_players()

            if 1 == len(active_players):
                explore_strategy = self._explore_strategy(state, Q_upper, Q_lower)
                self._explore_strategies[state] = explore_strategy

                if not self._exploit:
                    self._exploit_strategies[state] = { active_players[0]: self._explore_strategies[state] }
                else:
                    self._exploit_strategies[state] = { active_players[0]: self._exploit_strategy(state, Q_upper, Q_lower) }
            else:
                explore_strategy = self._solve_CCE(Q_upper, Q_lower)
                self._explore_strategies[state] = explore_strategy

                if self._exploit:
                    self._exploit_strategies[state] = { 
                        0: self._solve_row(Q_lower),
                        1: self._solve_row(-Q_upper.T)
                    }
                else:
                    self._exploit_strategies[state] = { 
                        0: np.sum(explore_strategy, axis=1), 
                        1: np.sum(explore_strategy, axis=0)
                    }

            # Update value functions
            V_upper[state] = np.sum(Q_upper * explore_strategy)
            V_lower[state] = np.sum(Q_lower * explore_strategy)

    def _model(self, state, actions, rewards, next_state):

        # If state has not been observed before, initialize the model
        if state not in self._counts:

            # Get action space
            action_shape = []

            for player_id in state.active_players():
                action_shape.append(state.num_actions(player_id))

            action_shape = tuple(action_shape)

            # Initialize model
            self._counts[state] = np.zeros(action_shape, dtype=int)
            self._rewards[state] = np.zeros(action_shape, dtype=float)
            self._transitions[state] = np.empty(action_shape, dtype=object)

            for idx in np.ndindex(action_shape):
                self._transitions[state][idx] = defaultdict(lambda: 0)
            
        # Update model
        actions = tuple(actions)
        self._counts[state][actions] += 1
        self._rewards[state][actions] += rewards[0]  # Only consider max-player rewards
        
        if next_state is not None:
            self._transitions[state][actions][next_state] += 1

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
            num_actions = row_actions * column_actions

            if state not in self._explore_strategies:
                strategy = np.ones((row_actions, column_actions), dtype=np.float64) / num_actions
            else:
                strategy = self._explore_strategies[state]

            action = np.random.choice(num_actions, p=strategy.flatten())
            row_action = action // column_actions
            column_action = action % column_actions

            return [row_action, column_action]
    
    def train(self):

        # Step counter
        total_steps = 0

        for _ in range(self._iteration_episodes):

            # Recompute strategy
            self._recompute()

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

                self._model(state, actions, rewards, next_state)

        # Return number of steps and episodes sampled, and episode statistics
        return total_steps, self._iteration_episodes, self._env.pull_stats()

    def strategy(self, state, player_id):
        if state not in self._exploit_strategies:
            num_actions = state.num_actions(player_id)
            return np.ones(num_actions, dtype=np.float64) / num_actions
        else:
            return self._exploit_strategies[state][player_id]
