from collections import defaultdict, deque
import numpy as np

from normal_form.solvers import build_solver


class Strategies:

    def __init__(self):
        self._strategies = defaultdict(dict)

    def add(self, state, player_id, strategy):
        self._strategies[player_id][state] = strategy
    
    def strategy(self, state, player_id):
        if player_id not in self._strategies:
            return None

        return self._strategies[player_id].get(state, None)


def get_payoffs(state, joint_action, V):
    payoffs = np.asarray(state.payoffs(joint_action), dtype=np.float64)
    successors = state.successors(joint_action)

    if len(successors) > 0:
        transitions = state.transitions(joint_action)

        for next_state, probability in zip(successors, transitions):
            payoffs += probability * V[next_state]

    return payoffs


class Solver:

    def __init__(self, config={}):
        lp_config = config.get("lp_config", {"method": "revised simplex"})
        self._solver = build_solver("linear_programming", lp_config)

    def __call__(self, game):

        # Initialize strategy
        strategies = Strategies()

        # Initialize value functions
        V = dict()

        # Enumerate state space
        state_space = set(game.initial_states())
        state_queue = deque(state_space)
        
        while len(state_queue) > 0:
            state = state_queue.popleft()

            # Add successors to queue
            for s in state.descendants():
                if s not in state_space:
                    state_queue.append(s)
                    state_space.add(s)
        
        # Sort states by depth
        state_space = sorted(state_space, key=lambda state: -state.depth())

        # Solve game in a depth-wise fashion using dynamic programming
        for state in state_space:
            active_players = state.active_players()

            # Determine whether this state is turn-based or simultaneous action
            if 1 == len(active_players):
                player_id = active_players[0]
                num_actions = state.num_actions(player_id)

                action_payoffs = []
                max_payoff = -np.infty

                for action in range(state.num_actions(player_id)):
                    payoffs = get_payoffs(state, [action], V)
                    action_payoffs.append(payoffs)

                    max_payoff = max(max_payoff, payoffs[player_id])
                    
                # Compute strategy and state values
                strategy = np.zeros(num_actions)
                row_value = 0
                column_value = 0
                norm = 0

                for action in range(num_actions):
                    if action_payoffs[action][player_id] == max_payoff:
                        strategy[action] = 1
                        row_value += action_payoffs[action][0]
                        column_value += action_payoffs[action][1]
                        norm += 1


                # Store strategy
                strategies.add(state, player_id, strategy / norm)

                # Store state values
                V[state] = np.asarray([row_value / norm, column_value / norm])

            else:

                # Build payoff matrix
                row_actions = state.num_actions(0)
                column_actions = state.num_actions(1)

                row_payoffs = np.zeros((row_actions,column_actions))
                column_payoffs = np.zeros((row_actions,column_actions))

                for row_action in range(row_actions):
                    for column_action in range(column_actions):
                        joint_action = [row_action, column_action]
                        payoffs = get_payoffs(state, joint_action, V)
                        row_payoffs[row_action, column_action] = payoffs[0]
                        column_payoffs[row_action, column_action] = payoffs[1]

                # Solve matrix game
                row_strategy, column_strategy, info = self._solver(row_payoffs, column_payoffs)

                # Get game values
                row_value = info["row_value"]
                column_value = info["column_value"]

                # Store strategies
                strategies.add(state, 0, row_strategy)
                strategies.add(state, 1, column_strategy)

                # Store values
                V[state] = np.asarray([row_value, column_value]) 

        return strategies, V
