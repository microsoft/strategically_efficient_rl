import math
import numpy as np
import scipy.signal

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.tune.utils import merge_dicts

tf = try_import_tf()

from algorithms.curiosity.modules import get_module_class

CURIOSITY_DEFAULTS = {
    # The curiosity module used to generate intrinsic reward
    "curiosity_module": "shaping",
    # Optional parameters for the curiosity module
    "curiosity_config": {},
    # Initial weight for intrinsic reward
    "start_weight": 1.0,
    # Final weight for intrinsic reward
    "end_weight": 0.0,
    # Steps over which weight should decay linearly (per worker)
    "exploration_steps": 500000,
    # Burn-in period (in time steps per worker) used to initialize mean and variance estimators
    "burn_in": 0,
    # Delay period (in time steps per worker) before mean and variance are computed
    "delay": 10000,
    # Normalization mechanism ("exponential", "total")
    "normalization": "exponential",
    # Decay constant for the explonentially weighted average
    "decay": 0.1,
}


class CuriosityModel(TFModelV2):

    # Initialization
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CuriosityModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        config = merge_dicts(CURIOSITY_DEFAULTS, model_config.get("custom_options", {}))

        # Reward normalization and dithering parameters
        self._gamma = model_config["intrinsic_gamma"]  # This may not be getting set properly
        self._reward_weight = config["start_weight"]
        self._end_weight = config["end_weight"]
        self._burn_in = config["burn_in"]
        self._delay = config["delay"]
        self._decay = config["decay"]
        self._normalization = config["normalization"]

        self._reward_step = max((self._reward_weight - self._end_weight) / config["exploration_steps"], 0.0)

        self._reward_mean = 0.0
        self._return_mean = 0.0
        self._return_variance = 1.0
        self._samples = 1e-6

        # Get curiosity config
        curiosity_config = config["curiosity_config"].copy()
        curiosity_config["intrinsic_gamma"] = model_config["intrinsic_gamma"]
        curiosity_config["num_other_agents"] = model_config["num_agents"] - 1

        # Build the curiosity module
        module_name = config["curiosity_module"]
        module_cls = get_module_class(module_name)

        self._curiosity_module = module_cls(obs_space, action_space, curiosity_config)
        self.register_variables(self._curiosity_module.variables())  # Can this be called more than once?

        # Initialize intrinsic loss dictionary
        self._intrinsic_losses = {}

        # Note that this class doesn't actually define a model output

    # Intrinsic reward - do reward normalization here - also, we don't want to normalize during evaluation - find a way to turn this off
    def intrinsic_reward(self, batch):
        rewards = self._curiosity_module.reward(batch)

        # No normalization
        if self._normalization is None:
            return self._reward_weight * rewards

        # Update sample counter
        count = float(len(rewards))
        self._samples += count

        # Decrement reward weights
        if self._reward_weight > self._end_weight:
            self._reward_weight -= self._reward_step

        # Return zero intrinsic reward (and don't update statistics) if we are still in the delay period
        if self._delay >= self._samples:
            return np.zeros_like(rewards)

        # Compute returns and update return statistics
        returns = scipy.signal.lfilter([1], [1, -self._gamma], rewards[::-1])[::-1]
        
        if "exponential" == self._normalization:
            # EXPONENTIALLY WEIGHTED VARIANCE
            delta = np.mean(returns) - self._return_mean
            self._reward_mean = self._decay * np.mean(rewards) + (1.0 - self._decay) * self._reward_mean
            self._return_mean += self._decay * delta
            self._return_variance = (1.0 - self._decay) * (self._return_variance + self._decay * (delta**2))
            deviation = math.sqrt(self._return_variance) + 1e-6
        else:
            # TOTAL VARIANCE
            delta = np.mean(returns) - self._return_mean
            self._reward_mean += np.mean(rewards) * (count / self._samples)
            self._return_mean += delta * (count / self._samples)
            self._return_variance += np.var(returns) * (count - 1)
            self._return_variance += (delta**2) * count * (self._samples - count) / self._samples
            deviation = math.sqrt(self._return_variance / (self._samples - 1)) + 1e-6

        # Return zero intrinsic reward if we are still in the burn-in period
        if self._burn_in >= self._samples:
            return np.zeros_like(rewards)

        # normalize and return rewards
        return self._reward_weight * (rewards - self._reward_mean) / deviation

    # The loss used to train the curiosity module
    def intrinsic_loss(self, train_batch):
        self._intrinsic_losses = self._curiosity_module.losses(train_batch)
        loss = None

        for value in self._intrinsic_losses.values():
            if loss is None:
                loss = value
            else:
                loss += value

        return loss

    # Statistics
    def metrics(self):
        metrics = super(CuriosityModel, self).metrics() or {}
        metrics.update(self._intrinsic_losses)

        return metrics
