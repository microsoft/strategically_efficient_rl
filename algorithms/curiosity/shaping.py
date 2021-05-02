# Task specific potential-based shaping
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

# Deals with TensorFlow's dynamic imports, which Ray doesn't like
tf = try_import_tf()


def dummy_potential(obs):
    return np.zeros(obs.shape[0])


SHAPING_DEFAULTS = {
    # The potential function used to generate the reward
    "potential": dummy_potential,
}


class PotentialShaping:

    def __init__(self, obs_space, action_space, curiosity_config):
        config = SHAPING_DEFAULTS.copy()
        config.update(curiosity_config)

        self._potential_fn = config["potential"]
        self._gamma = config["intrinsic_gamma"]

    def losses(self, loss_inputs):
        return {
            "intrinsic_dummy_loss": tf.constant(0.0)
        }

    def reward(self, batch):
        current_potential = self._potential_fn(batch[SampleBatch.CUR_OBS])
        next_potential = self._potential_fn(batch[SampleBatch.NEXT_OBS])

        return (self._gamma * next_potential) - current_potential

    def variables(self):
        return []
