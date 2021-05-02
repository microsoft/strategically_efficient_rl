from gym.spaces import Discrete, Box
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Context:

    def __init__(self, obs, permutation):
        self.obs = obs
        self.permutation = permutation


class ContextualRoShamBo(MultiAgentEnv):

    def __init__(self, env_config):
        transitive_actions = env_config.get("transitive_actions", 7)
        cyclic_actions = env_config.get("cyclic_actions", 3)
        num_contexts = env_config.get("num_contexts", 1)
        permute = env_config.get("permute", True)

        num_actions = cyclic_actions + transitive_actions
        assert (cyclic_actions % 2 == 1) or cyclic_actions == 0, "number of cyclic actions must be odd"

        # Build transitive payoff matrix
        self._G = np.zeros((num_actions, num_actions,))

        for i in range(num_actions):
            for j in range(num_actions):
                if i > j:
                    self._G[i, j] = 1.0
                elif i < j:
                    self._G[i, j] = 0.0
                else:
                    self._G[i, j] = 0.5

        # Build cycle payoff matrix
        for i in range(cyclic_actions):
            for j in range(cyclic_actions):
                if i == j:
                    value = 0.5
                elif ((i - j) % cyclic_actions) % 2 == 1:
                    value = 1.0
                else:
                    value = 0.0
                
                self._G[i + transitive_actions, j + transitive_actions] = value

        # Define observation and action spaces
        obs_space = Box(0.0, 1.0, shape=(num_contexts,))
        self.observation_space_dict = {"row": obs_space, "column": obs_space}

        action_space = Discrete(num_actions)
        self.action_space_dict = {"row": action_space, "column": action_space}

        # Define default observations
        self._dones = {"row": True, "column": True, "__all__": True}
        self._not_dones = {"row": False, "column": False, "__all__": False}
        self._infos = {"row": {}, "column": {}}

        # Define contexts and permutations
        self._contexts = []

        for idx in range(num_contexts):
            obs = np.zeros(num_contexts)
            obs[idx] = 1.0
            obs = {"row": obs, "column": obs}
            permutation = np.random.permutation(num_actions) if permute else np.arange(num_actions)
            self._contexts.append(Context(obs, permutation))
        
        self._current_context = None
    
    def reset(self):
        self._current_context = np.random.choice(self._contexts)
        return self._current_context.obs

    def step(self, action_dict):
        row_action = action_dict["row"]
        column_action = action_dict["column"]

        row_action = self._current_context.permutation[row_action]
        column_action = self._current_context.permutation[column_action]

        row_payoff = self._G[row_action, column_action]
        column_payoff = 1.0 - self._G[row_action, column_action]

        dones = self._dones

        rewards = {"row": row_payoff, "column": column_payoff}

        return self._current_context.obs, rewards, dones, self._infos

    def nash_conv(self, policy_dict):
        exploitability = 0.0
        row_values = 0.0
        column_values = 0.0    

        # Construct action batches
        row_actions = np.arange(self._G.shape[0])
        column_actions = np.arange(self._G.shape[1])

        for context in self._contexts:

            # Compute strategies - use the log-likelihood method provided by RLLib policies
            row_obs= [context.obs["row"]] * self._G.shape[0]
            column_obs = [context.obs["column"]] * self._G.shape[1]

            row_logits = policy_dict["row"].compute_log_likelihoods(row_actions, row_obs)
            column_logits = policy_dict["column"].compute_log_likelihoods(column_actions, column_obs)

            row_strategy = np.exp(row_logits)
            column_strategy = np.exp(column_logits)

            row_strategy /= np.sum(row_strategy)
            column_strategy /= np.sum(column_strategy)

            # Compute exploitabilities and values
            row_payoffs = self._G.dot(column_strategy)
            column_payoffs = row_strategy.dot(1.0 - self._G)

            row_values += 1.0 - np.max(column_payoffs)
            column_values += 1.0 - np.max(row_payoffs)

            exploitability += np.max(row_payoffs) + np.max(column_payoffs) - 1.0

        # Compute and return averages
        return {
            "nash_conv": exploitability / len(self._contexts),
            "row_value": row_values / len(self._contexts),
            "column_value": column_values / len(self._contexts),
        }
