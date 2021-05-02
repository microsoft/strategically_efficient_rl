from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class State(ABC):
    
    @abstractmethod
    def _key(self):
        pass

    @abstractmethod
    def depth(self):
        pass

    @abstractmethod
    def active_players(self):
        pass

    @abstractmethod
    def num_actions(self, player_id):
        pass

    @abstractmethod
    def payoffs(self, actions):
        pass

    @abstractmethod
    def successors(self, actions):
        pass

    @abstractmethod
    def transitions(self, actions):
        pass

    def descendants(self):
        action_shape = []

        for player_id in self.active_players():
            action_shape.append(self.num_actions(player_id))

        descendants = set()

        for action in np.ndindex(tuple(action_shape)):
            descendants.update(self.successors(action))

        return descendants

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return self._key() == other._key()

    def __ne__(self, other):
        return self._key() != other._key()
    
    def __repr__(self):
        return repr(self._key())


class Game(ABC):

    @abstractmethod
    def num_players(self):
        pass

    @abstractmethod
    def max_depth(self):
        pass

    @abstractmethod
    def max_payoff(self):
        pass

    @abstractmethod
    def initial_states(self):
        pass

    @abstractmethod
    def initial_distribution(self):
        pass
    
    def build_env(self):
        return Environment(self)


class EnvironmentState:

    def __init__(self, state):
        self._state = state

    def depth(self):
        return self._state.depth()

    def active_players(self):
        return self._state.active_players()

    def num_actions(self, player_id):
        return self._state.num_actions(player_id)

    def _key(self):
        return self._state._key()

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return self._key() == other._key()

    def __ne__(self, other):
        return self._key() != other._key()
    
    def __repr__(self):
        return repr(self._key())


class Environment:

    def __init__(self, game):
        self._game = game
        self._state = None

        self._stats = {}

    def reset(self):
        self._state = np.random.choice(self._game.initial_states(), p=self._game.initial_distribution())
        
        return EnvironmentState(self._state)

    def step(self, actions):
        self.get_stats(self._state, actions)

        payoffs = self._state.payoffs(actions)
        successors = self._state.successors(actions)

        if len(successors) == 0:
            return None, payoffs
        
        self._state = np.random.choice(successors, p=self._state.transitions(actions))
        
        return EnvironmentState(self._state), payoffs

    def num_players(self):
        return self._game.num_players()

    def max_depth(self):
        return self._game.max_depth()

    def max_payoff(self):
        return self._game.max_payoff()

    def get_stats(self, state, actions):
        pass

    def pull_stats(self):
        latest = self._stats
        self._stats = {}

        return latest
