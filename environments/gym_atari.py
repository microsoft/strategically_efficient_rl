import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from supersuit import frame_stack_v1, color_reduction_v0, resize_v0, normalize_obs_v0, dtype_v0


class GymAtari(MultiAgentEnv):
    """ This provides a multiagent-compatible wrapper for the gym atari environments, which 
        allows us tighter control of the preprocessing pipeline"""

    def __init__(self, env_config):
        env_config = env_config.copy()
        env_name = env_config.pop("game")
        frame_stack = env_config.pop("frame_stack", 4)
        color_reduction = env_config.pop("color_reduction", True)

        self._agent_id = env_config.pop("agent_id", "first_0")

        # Build Atari environment
        env = gym.make(env_name)

        if color_reduction:
            env = color_reduction_v0(env, mode='full')

        env = resize_v0(env, 84, 84)
        env = dtype_v0(env, dtype=np.float32)
        env = frame_stack_v1(env, frame_stack)
        self.env = normalize_obs_v0(env, env_min=0, env_max=1)

        # Get observation and action spaces
        self.observation_space_dict = {self._agent_id: self.env.observation_space}
        self.action_space_dict = {self._agent_id: self.env.action_space}

    def reset(self):
        return {self._agent_id: self.env.reset()}

    def step(self, action_dict):
        obs, reward, done, info = self.env.step(action_dict[self._agent_id])

        obs = {self._agent_id: obs}
        reward = {self._agent_id: reward}
        done = {self._agent_id: done}
        info = {self._agent_id: info}

        # Due to a quirk in RLLib, need to add a special '__all__' entry to the dones dict to terminate properly
        if all(done.values()):
            done["__all__"] = True
        else:
            done["__all__"] = False

        return obs, reward, done, info
