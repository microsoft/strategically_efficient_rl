import logging
import numpy as np

import ray
from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import ACTION_LOGP
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

from algorithms.agents.intrinsic import compute_advantages_intrinsic, \
    INTRINSIC_VALUE_TARGETS, INTRINSIC_VF_PREDS, INTRINSIC_REWARD
from algorithms.curiosity.curiosity_dense import CuriosityDense
from algorithms.curiosity.curiosity_vision import CuriosityVision

logger = logging.getLogger(__name__)


class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 intrinsic_value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 intrinsic_vf_preds,
                 curr_action_dist,
                 value_fn,
                 intrinsic_value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 intrinsic_head=True):
        """Constructs the loss for Proximal Policy Objective.
        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values
            intrinsic_value_targets (Placeholder): Placeholder for target intrinsic values
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for action prob output
                from the previous (before update) Model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from the previous (before update) Model evaluation.
            intrinsic_vf_preds (Placeholder): Placeholder for intrinsic value function output
                from the previous (before update) Model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            intrinsic_value_fn (Tensor): Current intrinsic value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Optional[tf.Tensor]): An optional bool mask of valid
                input elements (for max-len padded sequences (RNNs)).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            intrinsic_head (bool): If true, train the separate intrinsic value function.
        """
        if valid_mask is not None:

            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        else:

            def reduce_mean_valid(t):
                return tf.reduce_mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        vf_loss1 = tf.square(value_fn - value_targets)
        vf_clipped = vf_preds + tf.clip_by_value(
            value_fn - vf_preds, -vf_clip_param, vf_clip_param)
        vf_loss2 = tf.square(vf_clipped - value_targets)
        vf_loss = tf.maximum(vf_loss1, vf_loss2)
        self.mean_vf_loss = reduce_mean_valid(vf_loss)

        if intrinsic_head:
            intrinsic_vf_loss1 = tf.square(intrinsic_value_fn - intrinsic_value_targets)
            intrinsic_vf_clipped = intrinsic_vf_preds + tf.clip_by_value(
                intrinsic_value_fn - intrinsic_vf_preds, -vf_clip_param, vf_clip_param)
            intrinsic_vf_loss2 = tf.square(intrinsic_vf_clipped - intrinsic_value_targets)
            intrinsic_vf_loss = tf.maximum(intrinsic_vf_loss1, intrinsic_vf_loss2)
            vf_loss += intrinsic_vf_loss
            self.mean_intrinsic_vf_loss = reduce_mean_valid(intrinsic_vf_loss)

        loss = reduce_mean_valid(
            -surrogate_loss + cur_kl_coeff * action_kl +
            vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)

        self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[INTRINSIC_VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        train_batch[INTRINSIC_VF_PREDS],
        action_dist,
        model.value_function(),
        model.intrinsic_value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        intrinsic_head=policy.config["intrinsic_head"],
    )

    return policy.loss_obj.loss + model.intrinsic_loss(train_batch)


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "intrinsic_vf_loss": policy.loss_obj.mean_intrinsic_vf_loss,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def intrinsic_loss_stats(policy, batch, grad):
    return policy.model.metrics()


def vf_preds_and_logits_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        INTRINSIC_VF_PREDS: policy.model.intrinsic_value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
    }


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    # Determine if we are in evalaution mode - need to handle the loss initialization phase as well
    in_evaluation = policy.config["in_evaluation"] and policy.loss_initialized()

    # Add other agent's actions and value predictions to batch
    if not in_evaluation and policy.config["num_agents"] > 1:
        other_actions = []
        other_values = []

        if policy.loss_initialized():
            for other_policy, batch in other_agent_batches.values():
                other_actions.append(batch[SampleBatch.ACTIONS])

                if SampleBatch.VF_PREDS in batch:
                    other_values.append(batch[SampleBatch.VF_PREDS])
                else:
                    other_values.append(np.zeros_like(sample_batch[SampleBatch.VF_PREDS]))
        else:
            for _ in range(policy.config["num_agents"] - 1):
                other_actions.append(np.zeros_like(sample_batch[SampleBatch.ACTIONS]))
                other_values.append(np.zeros_like(sample_batch[SampleBatch.VF_PREDS]))

        sample_batch["other_actions"] = np.stack(other_actions, axis=-1)
        sample_batch["other_vf_preds"] = np.stack(other_values, axis=-1)

    # Compute intrinsic reward signal
    if not in_evaluation:
        if tf.executing_eagerly():
            sample_batch[INTRINSIC_REWARD] = policy.model.intrinsic_reward(sample_batch)
        else:
            with policy.get_session().as_default():
                sample_batch[INTRINSIC_REWARD] = policy.model.intrinsic_reward(sample_batch)

        if not policy.config["intrinsic_head"]:
            sample_batch[SampleBatch.REWARDS] += sample_batch[INTRINSIC_REWARD]

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = last_intrinsic_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r, last_intrinsic_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                                            sample_batch[SampleBatch.ACTIONS][-1],
                                            sample_batch[SampleBatch.REWARDS][-1],
                                            *next_state)

    if not in_evaluation and policy.config["intrinsic_head"]:
        batch = compute_advantages_intrinsic(
                    sample_batch,
                    last_r,
                    last_intrinsic_r,
                    policy.config["gamma"],
                    policy.config["intrinsic_gamma"],
                    policy.config["lambda"],
                    policy.config["intrinsic_lambda"])
    else:
        batch = compute_advantages(
                    sample_batch,
                    last_r,
                    policy.config["gamma"],
                    policy.config["lambda"])

    return batch


def clip_gradients(policy, optimizer, loss):
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.get_session())
        return self.kl_coeff_val


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model({
                SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                    [prev_action]),
                SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                    [prev_reward]),
                "is_training": tf.convert_to_tensor(False),
            }, [tf.convert_to_tensor([s]) for s in state],
                                        tf.convert_to_tensor([1]))
            return self.model.value_function()[0], self.model.intrinsic_value_function()[0]

        self._value = value


def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def make_model(policy, obs_space, action_space, config):
    model_config = config.get("model", {}).copy()
    model_config["intrinsic_gamma"] = config["intrinsic_gamma"]
    model_config["num_agents"] = config.get("num_agents", 1)

    # Get action distribution parameters
    logit_dim = policy.dist_class.required_model_output_shape(action_space, config)

    # Select class and build model
    if len(obs_space.shape) > 1:
        return CuriosityVision(obs_space, action_space, logit_dim, model_config, "default_model")
    else:
        return CuriosityDense(obs_space, action_space, logit_dim, model_config, "default_model")


PPOTFPolicy = build_tf_policy(
    name="PPOTFIntrinsicPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    grad_stats_fn=intrinsic_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    make_model=make_model,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])
