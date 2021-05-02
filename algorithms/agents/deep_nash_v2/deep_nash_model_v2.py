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
        vf_share_layers = model_config.get("vf_share_layers")
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

            policy_logits = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)

            average_policy_logits = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)
            
            if not vf_share_layers:
                last_layer = inputs

                for size in hiddens:
                    last_layer = tf.keras.layers.Dense(
                        size,
                        activation=activation,
                        kernel_initializer=normc_initializer(1.0))(last_layer)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
        
        intrinsic_value_out = tf.keras.layers.Dense(
            1,
            name="intrinsic_value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)

        self.base_model = tf.keras.Model(inputs, 
            [policy_logits, average_policy_logits, value_out, intrinsic_value_out])
        
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        if self._flatten:
            obs = input_dict["obs_flat"]
        else:
            obs = input_dict["obs"]

        policy_logits, self._average_policy_logits, self._value_out, self._intrinsic_value_out = self.base_model(obs)

        if self._exploration:
            outputs = policy_logits
        else:
            outputs = self._average_policy_logits

        return outputs, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def intrinsic_value_function(self):
        return tf.reshape(self._intrinsic_value_out, [-1])
    
    def average_policy_logits(self):
        return self._average_policy_logits
