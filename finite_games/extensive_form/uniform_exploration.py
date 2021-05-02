import numpy as np

from extensive_form.game_model import GameModel
from extensive_form.solver import Solver, Strategies


class UniformExploration:

    def __init__(self, env, config):
        self._env = env
        
        # Get configuration parameters
        self._iteration_episodes = config.get("iteration_episodes", 10)

        # Initialize game solver
        self._solver= Solver(config.get("solver_config", {}))

        # Initialize transition model
        self._model = GameModel(env.num_players(), 
                                env.max_payoff(),
                                min_samples=config.get("min_samples", 1), 
                                default_payoff=0)

        # Initialize strategy
        self._strategies = Strategies()

    def train(self):
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
                current_action = []

                for player_id in current_state.active_players():
                    current_action.append(np.random.choice(current_state.num_actions(player_id)))

                action_history.append(current_action)

                # Take joint action in environment
                current_state, payoffs = self._env.step(current_action)

                # Append payoffs to history
                reward_history.append(payoffs)

            total_steps += len(state_history)
            self._model.episode(state_history, action_history, reward_history)

        # Resolve the learned game
        self._strategies, _ = self._solver(self._model)

        # Return number of steps sampled
        return total_steps, self._iteration_episodes

    def strategy(self, state, player_id):
        strategy = self._strategies.strategy(state, player_id)

        if strategy is not None:
            return strategy
        else:
            num_actions = state.num_actions(player_id)
            return np.ones(num_actions) / num_actions
