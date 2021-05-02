from gym.spaces import Box, Discrete
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.utils import try_import_tf

# Deals with TensorFlow's dynamic imports, which Ray doesn't like
tf = try_import_tf()

OVP_DEFAULTS = {
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "relu",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 256],
    # Scale for network weight initialization
    "scale": 0.5,
}


class OpponentValuePrediction:

    def __init__(self, obs_space, action_space, curiosity_config):
        config = OVP_DEFAULTS.copy()
        config.update(curiosity_config)

        fcnet_activation = get_activation_fn(config["fcnet_activation"])
        fcnet_hiddens = config["fcnet_hiddens"]
        scale = config["scale"]
        num_other_agents = max(config["num_other_agents"], 1)

        # If we are in a visual domain like Atari, use initial convolutional layers
        if len(obs_space.shape) > 2:
            conv_activation = get_activation_fn(config["conv_activation"])
            conv_filters = config["conv_filters"]

            if conv_filters is None:
                conv_filters = _get_filter_config(obs_space.shape)

            inputs = tf.keras.layers.Input(shape=obs_space.shape)
            layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

            for out_size, kernel, stride in conv_filters[:-1]:
                layer = tf.keras.layers.Conv2D(
                                    out_size,
                                    kernel,
                                    strides=(stride, stride),
                                    activation=conv_activation,
                                    padding="same")(layer)

            out_size, kernel, stride = conv_filters[-1]
            layer = tf.keras.layers.Conv2D(
                                out_size,
                                kernel,
                                strides=(stride, stride),
                                activation=conv_activation,
                                padding="valid")(layer)

            layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(layer)
        else:
            inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape), ))
            layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)
        
        # Define fully connected layers
        for size in fcnet_hiddens:
            layer = tf.keras.layers.Dense(size, activation=fcnet_activation,
                                kernel_initializer=normc_initializer(scale))(layer)

        layer = tf.keras.layers.Dense(num_other_agents, activation=None,
                            kernel_initializer=normc_initializer(scale))(layer)

        # Define model
        self._model = tf.keras.Model(inputs=inputs, outputs=layer)

    def losses(self, loss_inputs):
        value_predictions = self._model(loss_inputs[SampleBatch.CUR_OBS])

        # Determine if we are in the single agent or multi-agent setting
        if "other_vf_preds" in loss_inputs:
            values = loss_inputs["other_vf_preds"]
        else:
            values = tf.expand_dims(loss_inputs[SampleBatch.VF_PREDS], axis=-1)
        
        squared_error = tf.square(value_predictions - tf.stop_gradient(values))

        return {
            "intrinsic_loss": tf.reduce_mean(squared_error)
        }

    def reward(self, batch):
        value_predictions = self._model.predict(batch[SampleBatch.CUR_OBS])

                # Determine if we are in the single agent or multi-agent setting
        if "other_vf_preds" in batch:
            values = batch["other_vf_preds"]
        else:
            values = np.expand_dims(batch[SampleBatch.VF_PREDS], axis=-1)

        return np.sqrt(np.mean(np.square(value_predictions - values), axis=-1))

    def variables(self):
        return self._model.variables
