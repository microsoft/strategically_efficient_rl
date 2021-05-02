from gym.spaces import Box, Discrete
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

RND_DEFAULTS = {
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "relu",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 256],
    # Number of outputs for the prediction model
    "fcnet_outputs": 512,
    # Scale for network weight initialization
    "scale": 0.5,
    # Whether to condition curiosity on the agent's action
    "agent_action": False,
    # Whether to condition curiosity on the joint action - overrides the joint_action field
    "joint_action": False,
}


class RandomNetworkDistillation:

    def __init__(self, obs_space, action_space, curiosity_config):
        config = RND_DEFAULTS.copy()
        config.update(curiosity_config)

        fcnet_activation = config["fcnet_activation"]
        fcnet_hiddens = config["fcnet_hiddens"]
        fcnet_outputs = config["fcnet_outputs"]
        scale = config["scale"]

        # Action conditional curiosity
        self._action_space = action_space
        self._agent_action = config["agent_action"]
        self._joint_action = config["joint_action"]
        self._num_other_agents = config["num_other_agents"]

        # If we are in a visual domain like Atari, use initial convolutional layers
        if len(obs_space.shape) > 2:
            conv_activation = get_activation_fn(config["conv_activation"])
            conv_filters = config["conv_filters"]

            if conv_filters is None:
                conv_filters = _get_filter_config(obs_space.shape)

            inputs = tf.keras.layers.Input(shape=obs_space.shape)

            target_layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)
            prediction_layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

            for out_size, kernel, stride in conv_filters[:-1]:
                target_layer = tf.keras.layers.Conv2D(
                                    out_size,
                                    kernel,
                                    strides=(stride, stride),
                                    activation=conv_activation,
                                    padding="same")(target_layer)
                prediction_layer = tf.keras.layers.Conv2D(
                                    out_size,
                                    kernel,
                                    strides=(stride, stride),
                                    activation=conv_activation,
                                    padding="same")(prediction_layer)

            out_size, kernel, stride = conv_filters[-1]
            target_layer = tf.keras.layers.Conv2D(
                                out_size,
                                kernel,
                                strides=(stride, stride),
                                activation=conv_activation,
                                padding="valid")(target_layer)
            prediction_layer = tf.keras.layers.Conv2D(
                                out_size,
                                kernel,
                                strides=(stride, stride),
                                activation=conv_activation,
                                padding="valid")(prediction_layer)

            target_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(target_layer)
            prediction_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(prediction_layer)
        else:
            inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape), ))

            target_layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)
            prediction_layer = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

        # Define action inputs
        if self._agent_action or self._joint_action:
            if isinstance(action_space, Discrete):
                action_input = tf.keras.layers.Input(shape=(action_space.n,))

                if self._joint_action:
                    other_action_input = tf.keras.layers.Input(shape=(self._num_other_agents * action_space.n,))

            elif isinstance(action_space, Box) and len(action_space.shape) == 1:
                action_input = tf.keras.layers.Input(shape=action_space.shape)

                if self._joint_action:
                    other_action_input = tf.keras.layers.Input(shape=(action_space.shape[0] * self._num_other_agents,))
            else:
                raise ValueError(f"Cannot handle action space: {action_space}")
            
            # Concatenate and actions to input list
            target_layer = tf.keras.layers.Concatenate()([target_layer, action_input])
            prediction_layer = tf.keras.layers.Concatenate()([prediction_layer, action_input])
            inputs = [inputs, action_input]

            if self._joint_action:
                target_layer = tf.keras.layers.Concatenate()([target_layer, other_action_input])
                prediction_layer = tf.keras.layers.Concatenate()([prediction_layer, other_action_input])
                inputs.append(other_action_input)

        # Define fully connected layers
        for size in fcnet_hiddens:
            target_layer = tf.keras.layers.Dense(size, activation=fcnet_activation,
                                kernel_initializer=normc_initializer(scale))(target_layer)
            prediction_layer = tf.keras.layers.Dense(size, activation=fcnet_activation,
                                kernel_initializer=normc_initializer(scale))(prediction_layer)

        target_layer = tf.keras.layers.Dense(fcnet_outputs, activation=None,
                            kernel_initializer=normc_initializer(scale))(target_layer)
        prediction_layer = tf.keras.layers.Dense(fcnet_outputs, activation=None,
                            kernel_initializer=normc_initializer(scale))(prediction_layer)

        # Define models - how do we get the variables from these models
        self._target_model = tf.keras.Model(inputs=inputs, outputs=target_layer)
        self._prediction_model = tf.keras.Model(inputs=inputs, outputs=prediction_layer)

    def losses(self, loss_inputs):
        inputs = [loss_inputs[SampleBatch.CUR_OBS]]

        if self._agent_action or self._joint_action:
            action = loss_inputs[SampleBatch.ACTIONS]

            if self._joint_action:
                other_action = tf.reshape(loss_inputs["other_actions"], (-1,))  # Something seriously wrong here!
            
            if isinstance(self._action_space, Discrete):
                action = tf.cast(action, tf.int32)
                action = tf.keras.backend.one_hot(action, num_classes=self._action_space.n)  # One-hot encode action for forward models

                if self._joint_action:
                    other_action = tf.cast(other_action, tf.int32)
                    other_action = tf.keras.backend.one_hot(other_action, num_classes=self._action_space.n)
                    other_action = tf.reshape(other_action, (-1, self._num_other_agents * self._action_space.n))
            elif self._joint_action:
                other_action = tf.reshape(other_action, (-1, self._num_other_agents * self._action_space.shape[0]))

            inputs.append(action)

            if self._joint_action:
                inputs.append(other_action)

        target = self._target_model(inputs)
        prediction = self._prediction_model(inputs)
        squared_error = tf.square(prediction - tf.stop_gradient(target))

        return {
            "intrinsic_loss": tf.reduce_mean(squared_error)
        }

    def reward(self, batch):
        inputs = [batch[SampleBatch.CUR_OBS]]

        if self._agent_action or self._joint_action:
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

            inputs.append(action)

            if self._joint_action:
                inputs.append(other_action)

        target = self._target_model.predict(inputs)
        prediction = self._prediction_model.predict(inputs)
        return np.mean(np.square(target - prediction), axis=-1)

    def variables(self):
        return self._target_model.variables + self._prediction_model.variables
