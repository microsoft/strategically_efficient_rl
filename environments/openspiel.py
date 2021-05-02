from gym.spaces import Box, Discrete
import numpy as np

from open_spiel.python.rl_environment import Environment
from open_spiel.python.policy import Policy
from open_spiel.python.algorithms.exploitability import nash_conv

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environments.reward_filter import RewardFilter


# Used for NashConv calculations
class RLLibPolicy(Policy):
    
    def __init__(self, game, policies):
        self._policies = policies
        super(RLLibPolicy, self).__init__(game, policies.keys())

    def _get_observation(self, state, player_id):
        if self.game.get_type().provides_information_state_tensor:
            return state.information_state_tensor(player_id)
        else:
            return state.observation_tensor(player_id)

    def action_probabilities(self, state, player_id=None):

        # Get player ID and legal actions
        if player_id is None:
            player_id = state.current_player()

        legal_actions = state.legal_actions(player_id)

        # Compute action logits for legal actions
        obs = self._get_observation(state, player_id)
        obs = [obs] * len(legal_actions)

        logits = self._policies[player_id].compute_log_likelihoods(legal_actions, obs)
        
        # Compute strategies
        probabilities = np.exp(logits)
        total_probability = np.sum(probabilities)

        if total_probability < 1:
            probabilities += (1 - total_probability) / len(legal_actions)

        probabilities /= np.sum(probabilities)

        # Build and return probability dict
        return {action: probability for action, probability in zip(legal_actions, probabilities)}


class OpenSpielGame(MultiAgentEnv):
    def __init__(self, env_config):
        env_config = env_config.copy()

        # Build reward filter - remove filtering parameters from config        
        self._reward_filter = RewardFilter(env_config.pop("reward_filter", {}))

        # Build underlying OpenSpiel game
        self._spielenv = Environment(**env_config)

        # Define the list of agent IDs
        self._agent_ids = list(range(self._spielenv.num_players))

        # Convert observation specification to the gym interface
        observation_spec = self._spielenv.observation_spec()

        if "info_state" in observation_spec:
            observation_space = Box(0.0, 1.0, shape=observation_spec["info_state"])
        else:
            raise Exception("OpenSpiel game does not define an observation space")

        self.observation_space_dict = self._make_dict([observation_space])
        self.observation_space = observation_space # Needed to run as a single-agent task

        # Convert action specification to the gym interface
        action_spec = self._spielenv.action_spec()

        if "num_actions" in action_spec:
            action_space = Discrete(action_spec["num_actions"])
        else:
            raise Exception("OpenSpiel game does not define a discrete action space")

        self.action_space_dict = self._make_dict([action_space])
        self.action_space = action_space # Needed to run as a single-agent task

        # Initialze current player and valid actions
        self._legal_actions = [[]] * self._spielenv.num_players

    def _make_dict(self, values):
        if len(values) == 1:
            return {agent_id: values[0] for agent_id in self._agent_ids}
        else:
            return dict(zip(self._agent_ids, values))

    def _make_dict_legal(self, values):
        dictionary = dict()

        if len(values) == 1:
            for agent_id, legal_actions in zip(self._agent_ids, self._legal_actions):
                if len(legal_actions) > 0:
                    dictionary[agent_id] = values[0]
        else:
            for agent_id, legal_actions, value in zip(self._agent_ids, self._legal_actions, values):
                if len(legal_actions) > 0:
                    dictionary[agent_id] = value

        return dictionary

    def reset(self):
        
        # Initialize reward filters
        self._reward_filter.reset()

        # Reset environment
        step = self._spielenv.reset()

        # Get the list of legal actions - must be done before we consruct the observation
        self._legal_actions = step.observations["legal_actions"]

        # Convert the observation list to a dictionary
        return self._make_dict_legal(step.observations["info_state"])

    def step(self, action_dict):

        # Convert action dictionary to list - only include actions for current player in turn-based games
        actions = []

        for agent_id, legal_actions in zip(self._agent_ids, self._legal_actions):
            if len(legal_actions) > 0:
                if action_dict[agent_id] not in legal_actions:
                    actions.append(np.random.choice(legal_actions))
                else:
                    actions.append(action_dict[agent_id])

        # Take a step in the environment
        step = self._spielenv.step(actions)

        # Updated list of legal actions - do this before constructing observations
        self._legal_actions = step.observations["legal_actions"]

        # Filter rewards - rewards may not be defined for the current state
        if step.rewards is None:
            rewards = [0.0] * self._spielenv.num_players
        else:
            rewards = self._reward_filter.filter(step.rewards, step.last())

        # If the game has ended, return observations and rewards for all agents
        if step.last():
            obs = self._make_dict(step.observations["info_state"])
            rewards = self._make_dict(rewards)
            dones = self._make_dict([True])
            dones["__all__"] = True
            infos = self._make_dict([{}])           
        else:
            obs = self._make_dict_legal(step.observations["info_state"])
            rewards = self._make_dict_legal(rewards)
            dones = self._make_dict_legal([False])
            dones["__all__"] = False
            infos = self._make_dict_legal([{}])

        return obs, rewards, dones, infos

    def nash_conv(self, policy_dict):
        policy = RLLibPolicy(self._spielenv.game, policy_dict)
        output = nash_conv(self._spielenv.game, policy, return_only_nash_conv=False)

        results = { "nash_conv": output.nash_conv }

        for agent_id, improvement in enumerate(output.player_improvements):
            results[f"player_{agent_id}"] = improvement

        return results
