from normal_form.solvers.linear_programming import LinearProgramming
from normal_form.solvers.hedge import Hedge

SOLVERS = {
    "linear_programming": LinearProgramming,
    "hedge": Hedge,
}


def build_solver(name, config):
    if name not in SOLVERS:
        raise ValueError(f"Solver '{name}' is not defined")

    return SOLVERS[name](config)