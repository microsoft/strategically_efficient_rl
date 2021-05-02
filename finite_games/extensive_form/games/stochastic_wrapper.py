from collections import defaultdict
import numpy as np

from extensive_form.games.extensive_game import State, Game


class StochasticState(State):
    
    def __init__(self, state, randomness):
        self._state = state

        # Get possible successors
        self._successors = list(state.descendants())

        # Get action space
        action_shape = []

        for player_id in self.active_players():
            action_shape.append(self.num_actions(player_id))

        # Build transitions probabilities
        action_probabilities = np.zeros(tuple(action_shape + [len(self._successors)]), dtype=np.float64)
        uniform_distribution = np.zeros((len(self._successors),), dtype=np.float64)

        for action in np.ndindex(tuple(action_shape)):
            successors = state.successors(action)
            transitions = state.transitions(action)

            dictionary = dict()

            for successor, transition in zip(successors, transitions):
                dictionary[successor] = transition
            
            for index, successor in enumerate(self._successors):
                action_probabilities[action, index] = dictionary[successor]
                uniform_distribution += dictionary[successor]

        # Normalize uniform action distribution
        uniform_distribution /= np.sum(uniform_distribution)

        # Compute transition probabilities
        self._transitions = np.zeros(tuple(action_shape + [len(self._successors)]), dtype=np.float64)

        for action in np.ndindex(tuple(action_shape)):
            self._transitions[action] = (1. - randomness) * action_probabilities[action]
            self._transitions[action] += randomness * uniform_distribution

    def depth(self):
        return self._state.depth()

    def active_players(self):
        return self._state.active_players()

    def num_actions(self, player_id):
        return self._state.num_actions(player_id)

    def payoffs(self, actions):
        return self._state.payoffs(actions)

    def successors(self, actions):
        return self._successors

    def transitions(self, actions):
        return self._transitions[actions]

    def descendants(self):
        return self._state.descendants()

    def __hash__(self):
        return hash(self._state)

    def __eq__(self, other):
        return self._state == other

    def __ne__(self, other):
        return self._state != other
    
    def __repr__(self):
        return repr(self._state)


class StochasticGame(Game):

    def __init__(self, game, randomness):
        self._game = game
        self._randomness = randomness


    def num_players(self):
        return self._game.num_players()

    def max_depth(self):
        return self._game.max_depth()

    def max_payoff(self):
        return self._game.max_payoff()

    @abstractmethod
    def initial_states(self):
        pass

    @abstractmethod
    def initial_distribution(self):
        return self._game.initial_distribution()
    
    def build_env(self):
        return Environment(self)