from collections import deque
import numpy as np


def get_payoff(player_id, state, action, V):
    payoff = state.payoffs(action)[player_id]
    successors = state.successors(action)

    if len(successors) > 0:
        successor_values = np.asarray([V[s] for s in successors])
        payoff += successor_values.dot(np.asarray(state.transitions(action)))

    return payoff


def compute_value(game, learner, player_id):

    # Identify the adversary
    adversary_id = 1 - player_id

    # Initialize value function
    V = dict()

    # Enumrate state space
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

    # Use dynamics programming to compute the value function
    for state in state_space:

        # Determine whether this is simultaneous action or turn-based (can be mixed within a game)
        if player_id not in state.active_players():
            
            # Get payoffs for each adversary action
            num_actions = state.num_actions(adversary_id)
            payoffs = [get_payoff(player_id, state, [action], V) for action in range(num_actions)]

            # Get minimum player payoff over adversary actions
            value = min(payoffs)

        elif adversary_id not in state.active_players():

            # Get payoffs for each player action
            num_actions = state.num_actions(player_id)
            payoffs = [get_payoff(player_id, state, [action], V) for action in range(num_actions)]
            
            # Get player strategy
            player_strategy = learner.strategy(state, player_id)

            # Compute expected value under player strategy
            value = player_strategy.dot(np.asarray(payoffs))

        else:

            # Build payoff matrix
            row_actions = state.num_actions(0)
            column_actions = state.num_actions(1)

            payoff_matrix = np.zeros((row_actions,column_actions))

            for row_action in range(row_actions):
                for column_action in range(column_actions):
                    joint_action = [row_action, column_action]
                    payoff_matrix[row_action, column_action] = get_payoff(player_id, state, joint_action, V)

            if 1 == player_id:
                payoff_matrix = payoff_matrix.T

            # Get player strategy
            player_strategy = learner.strategy(state, player_id)

            # Compute values for each adversary action
            payoffs = player_strategy.dot(payoff_matrix)

            # Compute minimum over adversary actions
            value = np.min(payoffs)

        # Add state value to 
        V[state] = value

    # Compute game value for initial states
    initial_values = np.asarray([V[s] for s in game.initial_states()])
    return initial_values.dot(game.initial_distribution())

def nash_conv(game, learner):
    
    # Compute best-response value for row and column players
    row_value = compute_value(game, learner, 0)
    column_value = compute_value(game, learner, 1)

    # Compute nash convergence value
    nash = game.max_payoff() - row_value - column_value

    return row_value, column_value, nash
