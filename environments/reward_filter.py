import numpy as np


def censor(rewards, rate):
    censored_rewards = []

    for reward in rewards:
        censored_rewards.append(reward if np.random.random() <= rate else 0.0)

    return censored_rewards


def normalize(rewards):
    assert len(rewards) == 2, "zero-sum reward is only applicable to two-player games"

    delta = (rewards[0] - rewards[1]) / 2.0
    return [delta, -delta]


class RewardFilter:

    def __init__(self, config):

        # Enable zero-sum reward - total reward is always zero
        self._zero_sum = config.get("zero_sum", False)

        # Enable random reward censoring
        self._censoring = config.get("censoring", None)

        # Enable reward accumulation
        self._accumulate = config.get("accumulate", False)
        self._totals = None

    def reset(self):
        self._totals = None

    def filter(self, rewards, done):
        
        # Normalize rewards if needed
        if self._zero_sum:
            rewards = normalize(rewards)

        # Censor rewards if needed
        if self._censoring is not None:
            rewards = censor(rewards, self._censoring)
        
        # Accumulate rewards as needed
        if self._accumulate:
            if self._totals is None:
                self._totals = np.asarray(rewards)
            else:
                self._totals += rewards

            rewards = self._totals if done else np.zeros_like(rewards)

        return rewards
