import numpy as np
import time

from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from multiagent import make_env  # NOTE: This seems to be a wrapper for our custom particle Env, not the base particle env, test that this can change

from environments.reward_filter import RewardFilter

class ParticleEnv(MultiAgentEnv):
    """
    This is a custom RLLib wrapper for the multi-agent particle environments which
    supports reward filtering and action concatenation
    """

    def __init__(self, env_config):
        env_config = env_config.copy()

        # Get reward filtering config
        self._reward_filter = RewardFilter(env_config.pop("reward_filter", {}))

        # Get team configs if needed
        self._teams = env_config.pop("teams", None)

        # Determine if we are using continuous or discrete actions
        self._discrete_action = ("discrete" == env_config.get("action_space", "discrete"))

        # Build base particle environment
        self._env = make_env(**env_config)

        # Calculate the number of agents
        if self._teams is None:
            self.num_agents = self._env.n
        else:
            self.num_agents = len(self._teams)
        
        # Generate a list of agent - not sure where this is actually used
        self.agent_ids = list(range(self.num_agents))

        # Build observation and action spaces
        if self._teams is None:
            self.observation_space_dict = self._make_dict(self._env.observation_space)
            self.action_space_dict = self._make_dict(self._env.action_space)
        else:
            obs_spaces =[]
            action_spaces = []

            for team in self._teams:
                size = np.sum([self._env.observation_space[idx].shape[0] for idx in team])
                lows = np.concatenate([self._env.observation_space[idx].low for idx in team])
                highs = np.concatenate([self._env.observation_space[idx].high for idx in team])
                obs_spaces.append(Box(np.min(lows), np.max(highs), shape=(size,)))   

                if self._discrete_action:
                    sizes = [self._env.action_space[idx].n for idx in team]
                    action_spaces.append(Discrete(np.prod(sizes)))
                else:
                    size = np.sum([self._env.action_space[idx].shape[0] for idx in team])
                    lows = np.concatenate([self._env.action_space[idx].low for idx in team])
                    highs = np.concatenate([self._env.action_space[idx].high for idx in team])
                    action_spaces.append(Box(np.min(lows), np.max(highs), shape=(size,)))

            self.observation_space_dict = self._make_dict(obs_spaces)
            self.action_space_dict = self._make_dict(action_spaces)

    def reset(self):
        self._reward_filter.reset()
        obs = self._env.reset()

        if self._teams is not None:
            obs = self._collapse_obs(obs)

        return self._make_dict(obs)

    def step(self, action_dict):
        actions = list(action_dict.values())

        if self._teams is not None:
            actions = self._expand_actions(actions)
            
        obs_list, rew_list, done_list, _ = self._env.step(actions)

        # Combine observations and rewards for teams
        if self._teams is not None:
            obs_list = self._collapse_obs(obs_list)
            rew_list, done_list = self._collapse_step(rew_list, done_list)

        # Filter rewards
        rew_list = self._reward_filter.filter(rew_list, any(done_list))

        obs_dict = self._make_dict(obs_list)
        rew_dict = self._make_dict(rew_list)
        done_dict = self._make_dict(done_list)
        done_dict["__all__"] = all(done_list)
        # FIXME: Currently, this is the best option to transfer agent-wise termination signal without touching RLlib code hugely.
        info_dict = self._make_dict([{"done": done} for done in done_list])

        return obs_dict, rew_dict, done_dict, info_dict

    def _collapse_obs(self, agent_obs):
        team_obs = []

        for team in self._teams:
            obs = [agent_obs[idx] for idx in team]
            team_obs.append(np.concatenate(obs))
        
        return team_obs

    def _collapse_step(self, agent_rewards, agent_dones):
        team_rewards = []
        team_dones = []

        for team in self._teams:
            team_rewards.append(np.sum([agent_rewards[idx] for idx in team]))
            team_dones.append(any([agent_dones[idx] for idx in team]))

        return team_rewards, team_dones

    def _expand_actions(self, team_actions):
        if self._discrete_action:
            return self._expand_discrete_actions(team_actions)

        agent_actions = [None] * self._env.n

        for team, actions in zip(self._teams, team_actions):
            actions = np.split(actions, len(team))

            for idx, agent in enumerate(team):
                agent_actions[agent] = actions[idx]
        
        return agent_actions

    def _expand_discrete_actions(self, team_actions):
        agent_actions = [None] * self._env.n

        for team, action in zip(self._teams, team_actions):
            actions = []

            for _ in range(len(team)):
                actions.append(action % 5)
                action = action // 5

            for idx, agent in enumerate(team):
                agent_actions[agent] = actions[idx]

        return agent_actions

    def _make_dict(self, values):
        return dict(zip(self.agent_ids, values))
