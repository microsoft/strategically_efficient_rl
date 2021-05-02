#!/usr/bin/env python3

'''
Simple benchmarking script to compare accuracy and speed of different game solvers.
'''

import time
import warnings

import numpy as np
import scipy.optimize

from normal_form.games import build_game


def normalize(strategy):
    strategy = np.clip(strategy, 0, 1)
    return strategy / np.sum(strategy)


def solve_row(G, solver="revised simplex"):
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
    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, solver)
    
    # Return row strategy
    return result.x[1:]


def solve_independent(G, solver="revised simplex"):
    row_strategy = solve_row(G, solver)
    column_strategy = solve_row(-G.T, solver)

    row_strategy = normalize(row_strategy)
    column_strategy = normalize(column_strategy)

    return row_strategy, column_strategy


def solve_joint(G, solver="revised simplex"):
    N, M = G.shape

    # Objective - maximize game value 'v'
    c = np.zeros(1 + N + M, dtype=np.float64)
    c[0] = -1.0

    # Ensure that both strategies guarantee payoff 'v' against all responses
    A_ub = np.zeros((M + N, 1 + N + M,), dtype=np.float64)
    A_ub[0:M, 0] = np.ones(M, dtype=np.float64)
    A_ub[M:M + N, 0] = -np.ones(N, dtype=np.float64)

    A_ub[0:M, 1:1 + N] = -G.T
    A_ub[M:M + N, 1 + N:1 + N + M] = G

    b_ub = np.zeros(M + N, dtype=np.float64)

    # Ensure that strategies are distributions
    A_eq = np.zeros((2, 1 + N + M,), dtype=np.float64)
    A_eq[0, 1:1 + N] = np.ones(N, dtype=np.float64)
    A_eq[1, 1 + N:1 + N + M] = np.ones(M, dtype=np.float64)
    b_eq = np.ones(2, dtype=np.float64)

    bounds = [(0.0,None,)] * (1 + N + M)
    bounds[0] = (None, None)

    # Use SciPy to solve the game
    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, solver)
    
    # Return row strategy
    return result.x[1:1 + N], result.x[1 + N:1 + N + M]


def solve_CCE(G, solver="revised simplex"):
    N, M = G.shape

    # There is no objective - just a satisfiability problem
    c = np.zeros(N * M, dtype=np.float64)
    
    # Ensure the solution is a probability distribution
    A_eq = np.ones((1, N * M,), dtype=np.float64)
    b_eq = np.ones(1, dtype=np.float64)
    bounds = [(0.0, None)] * (N * M)

    # Ensure joint strategy is a CCE
    b_ub = np.zeros(N + M, dtype=np.float64)
    
    rows = []

    # Row bounds
    for a in range(N):
        A_diff = G[a] - G
        rows.append(A_diff.flatten())

    # Column bounds
    for b in range(M):
        B_diff = (G.T - G.T[b]).T
        rows.append(B_diff.flatten())

    A_ub = np.stack(rows)
    
    # Solve program
    strategy = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, solver)
    strategy = strategy.x.reshape((N, M)) # TODO: This hasn't been tested

    # Compute marginals
    row_strategy = np.sum(strategy, axis=1)
    column_strategy = np.sum(strategy, axis=0)

    return row_strategy, column_strategy


def evaluate(name, solve_fn, solvers, games):
    print(f"\n\n===== Evaluating: {name} =====\n")

    for solver in solvers:

        # Solve games
        strategies = []
        start_time = time.process_time()

        for game in games:
            strategies.append(solve_fn(game.G, solver))

        total_time = time.process_time() - start_time

        # Compute NashConv error
        total_error = 0.0

        for game, (row_strategy, column_strategy) in zip(games, strategies):
            _, _, nash_conv = game.nash_conv(row_strategy, column_strategy)
            total_error += nash_conv
        
        print(f"{solver} - mean time: {total_time / len(games)}s, mean error: {total_error / len(games)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    solvers = ["revised simplex", "interior-point"]
    num_games = 50

    game = "zero_sum"
    rows = 11
    columns = 11
    config = {}

    games = [build_game(game, rows, columns, config) for _ in range(num_games)]

    print(f"\nGame: {game}, {num_games} instances")

    # Evaluate independent solver
    evaluate("independent", solve_independent, solvers, games)

    # Evaluate joint solver
    evaluate("joint", solve_joint, solvers, games)

    # Evaluate CCE solver
    evaluate("CCE", solve_CCE, solvers, games)
