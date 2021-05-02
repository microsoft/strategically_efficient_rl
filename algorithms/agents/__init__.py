# Importing this package should register the custom agents
from algorithms.agents.ppo.ppo import PPOTrainer  # Broken in 0.8.6
from algorithms.agents.deep_nash_v1.deep_nash_v1 import DeepNash as DeepNash_v1
from algorithms.agents.deep_nash_v2.deep_nash_v2 import DeepNash as DeepNash_v2

from ray.tune.registry import register_trainable

register_trainable("PPO_CURIOSITY", PPOTrainer)
register_trainable("DEEP_NASH_V1", DeepNash_v1)
register_trainable("DEEP_NASH_V2", DeepNash_v2)
