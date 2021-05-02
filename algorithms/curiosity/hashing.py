# Task specific potential-based shaping
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

# Deals with TensorFlow's dynamic imports, which Ray doesn't like
tf = try_import_tf()


def dummy_hash(obs, action):
    return np.zeros(obs.shape[0], dtype=np.int32)


HASHING_DEFAULTS = {
    # The hash function used to generate the counts
    "hash_fn": dummy_hash,
    # Number of cells in the hash table
    "hash_size": 0,
    # Scale factor for the counts
    "scale": 1.0,
}


class Hashing:

    def __init__(self, obs_space, action_space, curiosity_config):
        config = HASHING_DEFAULTS.copy()
        config.update(curiosity_config)

        self._scale = config["scale"]
        self._hash_fn = config["hash_fn"]
        
        hash_size = config["hash_size"]

        if hash_size > 0:
            self._counts = np.ones(shape=(hash_size,), dtype=np.float)
        else:
            self._counts = None

    def losses(self, loss_inputs):
        return {
            "intrinsic_dummy_loss": tf.constant(0.0)
        }

    def reward(self, batch):
        obs = batch[SampleBatch.CUR_OBS]
        actions = batch[SampleBatch.ACTIONS]

        if self._counts is None:
            return np.zeros_like(obs)

        # Compute hash function for all observations
        hashes = self._hash_fn(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        
        # Get counts
        counts = np.take(self._counts, hashes)
        rewards = 1.0 / np.sqrt(self._scale * counts)

        # Increment counts
        counts += 1.0
        np.put(self._counts, hashes, counts)

        return rewards

    def variables(self):
        return []
