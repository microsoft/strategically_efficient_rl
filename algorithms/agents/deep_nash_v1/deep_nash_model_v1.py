import numpy as np

from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

from algorithms.curiosity import CuriosityModel


class NashModel(CuriosityModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, exploration=True):
        super(NashModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self._exploration = exploration

        # Use this parameter to share layers between the Q and policy networks
        share_layers = model_config.get("share_layers")
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens")

        # Build input and feature layers
        if len(obs_space.shape) > 1:
            self._flatten = False
            raise NotImplementedError
        else:
            self._flatten = True
            inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape), ), name="observations")
            last_layer = inputs

            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)

            q_outputs = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)
            
            if not share_layers:
                last_layer = inputs

                for size in hiddens:
                    last_layer = tf.keras.layers.Dense(
                        size,
                        activation=activation,
                        kernel_initializer=normc_initializer(1.0))(last_layer)
                
            policy_logits = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)

        self.base_model = tf.keras.Model(inputs, [q_outputs, policy_logits]) 
        self.register_variables(self.base_model.variables)

        self.learning_rate = tf.Variable(tf.constant(0.0), trainable=False, dtype=tf.float32)

    def forward(self, input_dict, state, seq_lens):
        if self._flatten:
            obs = input_dict["obs_flat"]
        else:
            obs = input_dict["obs"]

        self._q_values, self._policy_logits = self.base_model(obs)

        # Compute policy
        policy_weights = tf.exp(self._policy_logits)
        policy_weights = policy_weights - tf.reduce_mean(policy_weights, axis=-1)
        policy_weights = tf.clip_by_value(policy_weights, 1e-8, 1e8)
        self._policy = policy_weights / tf.reduce_sum(policy_weights, axis=-1)

        # Compute Q estimates
        if self._exploration:
            outputs = self.learning_rate * self._q_values
        else:
            outputs = self._policy_logits

        return outputs, state

    def q_values(self):
        return self._q_values

    def average_policy(self):
        return self._policy
    
    def average_policy_logits(self):
        return self._policy_logits
