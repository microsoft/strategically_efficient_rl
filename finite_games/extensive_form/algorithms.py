import numpy as np

from extensive_form.optimistic_ulcb import OptimisticULCB
from extensive_form.strategic_ulcb import StrategicULCB
from extensive_form.optimistic_nash_q import OptimisticNashQ
from extensive_form.optimistic_nash_v import OptimisticNashV
from extensive_form.uniform_exploration import UniformExploration
from extensive_form.optimistic_q_learning import OptimisticQLearning
from extensive_form.nash_q import NashQ
from extensive_form.strategic_nash_q import StrategicNashQ


class UniformRandom:

    def __init__(self, env, config):
        self._env = env

    def sample(self):
        state = self._env.reset()
        steps = 0

        while state is not None:
            action = []

            for player_id in state.active_players():
                action.append(np.random.randint(0, state.num_actions(player_id)))
            
            state = self._env.step(action)
            steps += 1

        return steps

    def strategy(self, state, player_id):
        num_actions = state.num_actions(player_id)
        return np.ones(num_actions) / num_actions


ALGORITHMS = {
    "random": UniformRandom,
    "uniform_exploration": UniformExploration,
    "optimistic_nash_q": OptimisticNashQ,
    "optimistic_nash_v": OptimisticNashV,
    "optimistic_q_learning": OptimisticQLearning,
    "nash_q": NashQ,
    "strategic_nash_q": StrategicNashQ,
    "optimistic_ulcb": OptimisticULCB,
    "strategic_ulcb": StrategicULCB,
}

def build_algorithm(alg, env, config={}):
    if alg not in ALGORITHMS:
        raise ValueError(f"Leanring algorithm '{alg}' is not defined")

    return ALGORITHMS[alg](env, config)
