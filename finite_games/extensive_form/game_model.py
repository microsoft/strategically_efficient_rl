from collections import defaultdict
import numpy as np

from extensive_form.games.extensive_game import State, Game


class ActionModel:

    def __init__(self):
        self._successors = []
        self._counts = defaultdict(lambda: 1)

    def successors(self):
        return self._successors

    def transitions(self):
        transitions = []

        for state in self._successors:
            transitions.append(self._counts[state])

        return np.asarray(transitions, dtype=np.float64) / np.sum(transitions)
    
    def observe(self, next_state):
        if next_state not in self._counts:
            self._successors.append(next_state)
        
        self._counts[next_state] += 1


class StateModel(State):
    
    def __init__(self, state, num_players, min_samples, default_payoff):
        self._state = state
        self._min_samples = min_samples
        self._default_payoff = default_payoff

        # Compute action space
        action_shape = []

        for player_id in state.active_players():
            action_shape.append(state.num_actions(player_id))

        action_shape = tuple(action_shape)

        # Initialize action counters
        self._counts = np.zeros(action_shape)

        # Initialize payoff estimates
        self._payoffs = np.zeros(action_shape + (num_players,))

        # Initialize transition models
        self._actions = np.zeros(action_shape, dtype=object)

        for idx in np.ndindex(action_shape):
            self._actions[idx] = ActionModel()

    def _key(self):
        return self._state._key()

    def depth(self):
        return self._state.depth()

    def active_players(self):
        return self._state.active_players()

    def num_actions(self, player_id):
        return self._state.num_actions(player_id)

    def payoffs(self, actions):
        actions = tuple(actions)
        payoffs = self._payoffs[actions]

        if self._counts[actions] < self._min_samples:
            return np.full_like(payoffs, self._default_payoff, dtype=np.float64)
        else:
            return payoffs / self._counts[actions]

    def successors(self, actions):
        actions = tuple(actions)

        if self._counts[actions] >= self._min_samples:
            return self._actions[actions].successors()
        else:
            return []

    def transitions(self, actions):
        actions = tuple(actions)

        if self._counts[actions] >= self._min_samples:
            return self._actions[actions].transitions()
        else:
            return []

    def descendants(self):
        descendants = set()

        for idx in np.ndindex(self._actions.shape):
            if self._counts[idx] >= self._min_samples:
                descendants.update(self._actions[idx].successors())
            
        return descendants

    def observe(self, actions, payoffs, next_state):
        actions = tuple(actions)

        if self._counts[actions] < self._min_samples:
            self._counts[actions] += 1
            self._payoffs[actions] += np.asarray(payoffs, dtype=np.float64)

            if next_state is not None:
                self._actions[actions].observe(next_state)


class GameModel(Game):

    def __init__(self, num_players, max_payoff, min_samples=1, default_payoff=0):
        self._num_players = num_players
        self._max_payoff = max_payoff
        self._min_samples = min_samples
        self._default_payoff = default_payoff

        # Initialize state space
        self._states = dict()

        # Define initial states
        self._initial_states = []
        self._initial_counts = defaultdict(lambda: 1)

    def num_players(self):
        return self._num_players

    def max_payoff(self):
        return self._max_payoff

    def initial_states(self):
        if 0 == len(self._initial_states):
            raise Exception("Model not initialized, no episodes have been observed")

        return self._initial_states

    def initial_distribution(self):
        initial_distribution = []

        for state in self._initial_states:
            initial_distribution.append(self._initial_counts[state])

        return np.asarray(initial_distribution, dtype=np.float64) / np.sum(initial_distribution)
    
    def episode(self, states, actions, rewards):
        if len(states) > 0:

            # Process initial state
            initial_state = states[0]

            if initial_state not in self._states:
                self._states[initial_state] = StateModel(initial_state, 
                                                         self._num_players, 
                                                         self._min_samples, 
                                                         self._default_payoff)

            if initial_state not in self._initial_counts:
                self._initial_states.append(self._states[initial_state])
            
            self._initial_counts[initial_state] += 1

            # Observe state transitons - we never observe the payoff of the final state
            for idx in range(len(states)):

                if idx + 1 < len(states):
                    next_state = states[idx + 1]

                    if next_state not in self._states:
                        self._states[next_state] = StateModel(next_state, 
                                                    self._num_players, self._min_samples, self._default_payoff)
                    
                    next_state = self._states[next_state]
                else:
                    next_state = None
                
                self._states[states[idx]].observe(actions[idx], rewards[idx], next_state)
        