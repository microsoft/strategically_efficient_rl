from collections import defaultdict
import numpy as np
import scipy.optimize


class StrategicULCB:

    def __init__(self, env, config):
        self._env = env

        # Check that we are in a two-player game
        assert env.num_players() == 2, "Strategic ULCB is only implemented for 2-player games"

        # Get configuration
        self._iteration_episodes = config.get("iteration_episodes", 10)
        self._horizon = config.get("horizon", env.max_depth())
        self._beta = config.get("beta", 0.0)
        self._solver = config.get("solver", "revised simplex")

        # Set initial value - allows us to override optimistic initialization
        self._initial_value = config.get("initial_value", env.max_payoff()) # Is this properly defined by the environment?

        # initialize game model
        self._counts = dict()
        self._transitions = dict()
        self._rewards = dict()

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

        # Initialize min-max estimates
        Q_max = -np.Inf
        Q_min = np.Inf

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
                if self._counts[state][idx] > 0:  # Do we ever increment the counts?
                    
                    # Compute exploration bonus
                    beta = self._beta / np.sqrt(self._counts[state][idx])

                    # Compute average reward
                    reward = self._rewards[state][idx] / self._counts[state][idx]

                    # Get mean value of next state
                    next_upper = 0.
                    next_lower = 0.

                    # Is this implemented correctly
                    if self._transitions[state][idx] is not None:
                        for next_state, count in self._transitions[state][idx].items():
                            weight = float(count) / float(self._counts[state][idx])
                            next_upper += weight * V_upper[next_state]
                            next_lower += weight * V_lower[next_state]

                    # Set Q bounds
                    Q_upper[idx] = reward + next_upper + beta
                    Q_lower[idx] = reward + next_lower - beta
                else:
                    Q_upper[idx] = self._initial_value
                    Q_lower[idx] = 0.

            Q_max = max(Q_max, np.max(Q_upper))
            Q_min = min(Q_min, np.min(Q_lower))

            Q_upper[idx] = min(Q_upper[idx], self._initial_value)
            Q_lower[idx] = max(Q_lower[idx], 0.)

            # Update value functions
            active_players = state.active_players()

            if 1 == len(active_players):

                # Update strategies
                self._explore_strategies[state] = self._explore_strategy(state, Q_upper, Q_lower)
                self._exploit_strategies[state] = { active_players[0]: self._exploit_strategy(state, Q_upper, Q_lower) }

                # Update value functions - modified for a strategic update
                if 0 == active_players[0]:
                    V_upper[state] = np.max(Q_upper)
                    V_lower[state] = np.max(Q_lower) # Lower bound can only be larger
                else:
                    V_upper[state] = np.min(Q_upper) # Upper bound can only be smaller
                    V_lower[state] = np.min(Q_lower)
            else:
                row_explore, column_exploit = self._solve_nash(Q_upper)
                row_exploit, column_explore = self._solve_nash(Q_lower)

                self._explore_strategies[state] = {
                    active_players[0]: row_explore,
                    active_players[1]: column_explore
                }

                self._exploit_strategies[state] = {
                    active_players[0]: row_exploit,
                    active_players[1]: column_exploit
                }

        return Q_min, Q_max

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
        self._rewards[state][actions] += rewards[0]  # Only consider max-player rewards - did we forget to normalize this?
        
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

        # Initialize min-max estimates
        Q_max = -np.Inf
        Q_min = np.Inf

        # Step counter
        total_steps = 0

        for _ in range(self._iteration_episodes):

            # Recompute strategy
            q_min, q_max = self._recompute()

            Q_min = min(Q_min, q_min)
            Q_max = max(Q_max, q_max)

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
        stats = self._env.pull_stats()
        stats["q_min"] = Q_min
        stats["q_max"] = Q_max

        return total_steps, self._iteration_episodes, stats

    def strategy(self, state, player_id):
        if state not in self._exploit_strategies:
            num_actions = state.num_actions(player_id)
            return np.ones(num_actions, dtype=np.float64) / num_actions
        else:
            return self._exploit_strategies[state][player_id]
