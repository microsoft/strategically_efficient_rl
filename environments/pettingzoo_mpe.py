import importlib
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from supersuit import dtype_v0


class PettingZooMPE(MultiAgentEnv):

    def __init__(self, env_config):
        env_config = env_config.copy()
        env_name = env_config.pop("scenario")

        # Get random agents
        self._random_agents = env_config.pop("random_agents", [])

        # Load appropriate PettingZoo class
        env_module = importlib.import_module("pettingzoo.mpe." + env_name)

        # Build PettingZoo environment
        env = env_module.parallel_env(**env_config)  # Has to be the parallel environment due to an aparent bug
        self.env = dtype_v0(env, dtype=np.float32)

        # Get observation and action spaces - just changes the names to be compatible with RLLib
        self.observation_space_dict = self._remove_fixed(self.env.observation_spaces)
        self.action_space_dict = self._remove_fixed(self.env.action_spaces)

    def _remove_fixed(self, dictionary):
        dictionary = dictionary.copy()

        for agent_id in self._random_agents:
            dictionary.pop(agent_id)

        return dictionary

    def _add_fixed(self, dictionary):
        dictionary = dictionary.copy()

        for agent_id in self._random_agents:
            dictionary[agent_id] = np.random.randint(0, self.env.action_spaces[agent_id].n)

        return dictionary

    def reset(self):
        return self._remove_fixed(self.env.reset())

    def step(self, action_dict):
        action_dict = self._add_fixed(action_dict)

        obs, reward, done, info = self.env.step(action_dict)

        obs = self._remove_fixed(obs)
        reward = self._remove_fixed(reward)
        done = self._remove_fixed(done)
        info = self._remove_fixed(info)

        # Due to a quirk in RLLib, need to add a special '__all__' entry to the dones dict to terminate properly
        if all(done.values()):
            done["__all__"] = True
        else:
            done["__all__"] = False

        return obs, reward, done, info
