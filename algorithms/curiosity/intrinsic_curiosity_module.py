from gym.spaces import Box, Discrete
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.utils import try_import_tf

# Deals with TensorFlow's dynamic imports, which Ray doesn't like
tf = try_import_tf()

# Defaults match (Pathak et. al. 2017) for visual environments
ICM_DEFAULTS = {
    # The size of the latent feature space
    "latent_state_size": 256,
    # Scale for network weight initialization
    "scale": 0.5,
    # Nonlinearity for the encoder convnet
    "conv_activation": "elu",
    # Convolutional fliters for the encoder. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for the dense layers of the encoder
    "encoder_activation": "relu",
    # Sizes of the dense layers for the encoder
    "encoder_hiddens": [32, 32],
    # Nonlinearity for the forward model
    "forward_activation": "relu",
    # Sizes of the dense layers for the forward model
    "forward_hiddens": [256],
    # Nonlinearity for the inverse model
    "inverse_activation": "relu",
    # Sizes of the dense layers for the inverse model
    "inverse_hiddens": [256],
    # If true, don't incorporate the forward loss into the encoder loss
    "stop_gradient": True,
    # Whether to condition curiosity on the joint action - overrides the joint_action field
    "joint_action": False,
}


def _build_encoder(obs_space, action_space, config):
    scale = config["scale"]
    latent_state_size = config["latent_state_size"]

    # If we are in a visual domain like Atari, use initial convolutional layers
    if len(obs_space.shape) > 2:
        conv_activation = get_activation_fn(config["conv_activation"])
        conv_filters = config["conv_filters"]

        if conv_filters is None:
            conv_filters = _get_filter_config(obs_space.shape)

        inputs = tf.keras.layers.Input(shape=obs_space.shape)
        layer = inputs

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
        layer = tf.keras.layers.Conv2D(
                            latent_state_size,
                            [1, 1],
                            strides=(stride, stride),
                            activation=conv_activation,
                            padding="same")(layer)
        outputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(layer)
    else:
        activation = get_activation_fn(config["encoder_activation"])
        hiddens = config["encoder_hiddens"]

        inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape),))
        layer = inputs

        for size in hiddens:
            layer = tf.keras.layers.Dense(size, activation=activation,
                                kernel_initializer=normc_initializer(scale))(layer)

        outputs = tf.keras.layers.Dense(latent_state_size, activation=None,
                            kernel_initializer=normc_initializer(scale))(layer)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _build_forward_model(obs_space, action_space, config):
    scale = config["scale"]
    latent_state_size = config["latent_state_size"]
    activation = get_activation_fn(config["forward_activation"])
    hiddens = config["forward_hiddens"]
    joint_action = config["joint_action"]
    num_other_agents = config["num_other_agents"]

    obs = tf.keras.layers.Input(shape=(latent_state_size, ))

    if isinstance(action_space, Discrete):
        action = tf.keras.layers.Input(shape=(action_space.n,))
        
        if joint_action:
            other_action = tf.keras.layers.Input(shape=(num_other_agents * action_space.n, ))
    elif isinstance(action_space, Box) and len(action_space.shape) == 1:
        action = tf.keras.layers.Input(shape=action_space.shape[0])

        if joint_action:
            other_action = tf.keras.layers.Input(shape=(action_space.shape[0] * num_other_agents,))
    else:
        raise ValueError(f"Cannot handle action space: {action_space}")

    inputs = [obs, action]

    if joint_action:
        inputs.append(other_action)

    layer = tf.keras.layers.concatenate(inputs, axis=-1)

    for size in hiddens:
        layer = tf.keras.layers.Dense(size, activation=activation,
                            kernel_initializer=normc_initializer(scale))(layer)

    output = tf.keras.layers.Dense(latent_state_size, activation=None,
                            kernel_initializer=normc_initializer(scale))(layer)

    return tf.keras.Model(inputs=inputs, outputs=output)


def _build_inverse_model(obs_space, action_space, config):
    scale = config["scale"]
    latent_state_size = config["latent_state_size"]
    activation = get_activation_fn(config["inverse_activation"])
    hiddens = config["inverse_hiddens"]
    joint_action = config["joint_action"]
    num_other_agents = config["num_other_agents"]

    current_obs = tf.keras.layers.Input(shape=(latent_state_size, ))
    next_obs = tf.keras.layers.Input(shape=(latent_state_size, ))

    layer = tf.keras.layers.concatenate([current_obs, next_obs], axis=-1)

    for size in hiddens:
        layer = tf.keras.layers.Dense(size, activation=activation,
                            kernel_initializer=normc_initializer(scale))(layer)

    if isinstance(action_space, Discrete):
        outputs = [tf.keras.layers.Dense(action_space.n, 
                            activation=None, kernel_initializer=normc_initializer(scale))(layer)]
        if joint_action:
            outputs.append(tf.keras.layers.Dense(action_space.n * num_other_agents, 
                                    activation=None, kernel_initializer=normc_initializer(scale))(layer))
    elif isinstance(action_space, Box) and len(action_space.shape) == 1:
        outputs = [tf.keras.layers.Dense(action_space.shape[0], 
                            activation=None, kernel_initializer=normc_initializer(scale))(layer)]
        if joint_action:
            outputs.append(tf.keras.layers.Dense(action_space.shape[0] *  num_other_agents, 
                                activation=None, kernel_initializer=normc_initializer(scale))(layer))
    else:
        raise ValueError(f"Cannot handle action space: {action_space}")

    return tf.keras.Model(inputs=[current_obs, next_obs], outputs=outputs)


class IntrinsicCuriosityModule:

    def __init__(self, obs_space, action_space, curiosity_config):
        config = ICM_DEFAULTS.copy()
        config.update(curiosity_config)

        # Capture loss configuration
        self._action_space = action_space
        self._stop_gradient = config["stop_gradient"]
        self._joint_action = config["joint_action"]
        self._num_other_agents = config["num_other_agents"]

        # Define models
        self._encoder = _build_encoder(obs_space, action_space, config)
        self._forward_model = _build_forward_model(obs_space, action_space, config)
        self._inverse_model = _build_inverse_model(obs_space, action_space, config)

    def losses(self, loss_inputs):
        cur_state = self._encoder(loss_inputs[SampleBatch.CUR_OBS])
        next_state = self._encoder(loss_inputs[SampleBatch.NEXT_OBS]) # This isn't defined! <- This comment seems wrong, check on this?
        action = loss_inputs[SampleBatch.ACTIONS]

        # Define inverse model loss
        if self._joint_action:
            action_outputs, other_action_outputs = self._inverse_model([cur_state, next_state])
            other_action = tf.reshape(loss_inputs["other_actions"], (-1,))
        else:
            action_outputs = self._inverse_model([cur_state, next_state])

        if isinstance(self._action_space, Discrete):
            action = tf.cast(action, tf.int32)
            inverse_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=action_outputs)
            action = tf.keras.backend.one_hot(action, num_classes=self._action_space.n)  # One-hot encode action for forward models

            if self._joint_action:
                other_action = tf.cast(other_action, tf.int32)
                other_action_outputs = tf.reshape(other_action_outputs, (-1, self._action_space.n))
                inverse_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=other_action, logits=other_action_outputs)
                other_action = tf.keras.backend.one_hot(other_action, num_classes=self._action_space.n)
                other_action = tf.reshape(other_action, (-1, self._num_other_agents * self._action_space.n))
        else:
            inverse_loss = tf.reduce_mean(tf.square(action - action_outputs))

            if self._joint_action:
                inverse_loss += tf.reduce_mean(tf.square(other_action - tf.reshape(other_action_outputs, (-1, self._action_space.shape[0]))))
                other_action = tf.reshape(other_action, (-1, self._num_other_agents * self._action_space.shape[0]))

        # Stop encoder gradients if necessary
        if self._stop_gradient:
            cur_state = tf.stop_gradient(cur_state)
            next_state = tf.stop_gradient(next_state)

        # Define forward model loss
        forward_inputs = [cur_state, action]

        if self._joint_action:
            forward_inputs.append(other_action)

        forward_outputs = self._forward_model(forward_inputs)
        forward_loss = tf.reduce_mean(tf.square(forward_outputs - next_state))

        return {
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
        }

    def reward(self, batch):
        cur_state = self._encoder.predict(batch[SampleBatch.CUR_OBS])
        next_state = self._encoder.predict(batch[SampleBatch.NEXT_OBS])
        action = batch[SampleBatch.ACTIONS]

        if self._joint_action:
            other_action = batch["other_actions"]

        # One-hot encode actions if needed
        if isinstance(self._action_space, Discrete):
            eye = np.eye(self._action_space.n)
            action = eye[action]  # One-hot encode action for forward models

            if self._joint_action:
                other_action = eye[other_action.reshape(-1)]
                other_action = other_action.reshape((-1, self._num_other_agents * self._action_space.n))
        elif self._joint_action:
            other_action = other_action.reshape((-1, self._num_other_agents * self._action_space.shape[0]))

        if self._joint_action:
            prediction = self._forward_model.predict([cur_state, action, other_action])
        else:
            prediction = self._forward_model.predict([cur_state, action])

        return np.mean(np.square(prediction - next_state), axis=-1)

    def variables(self):
        return self._encoder.variables + self._forward_model.variables + self._inverse_model.variables
