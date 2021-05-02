import logging
import numpy as np
import scipy.signal

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
from algorithms.agents.deep_nash_v2.deep_nash_model_v2 import NashModel

logger = logging.getLogger(__name__)

CURRENT_ACTION_MASK = "current_action_mask"


class AveragePPOLoss:
    def __init__(self,
                 curr_action_dist,
                 prev_dist,
                 value_targets,
                 intrinsic_value_targets,
                 advantages,
                 actions,
                 prev_actions_logp,
                 vf_preds,
                 intrinsic_vf_preds,
                 value_fn,
                 intrinsic_value_fn,
                 curr_action_mask,
                 average_action_dist,
                 cur_kl_coeff,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 average_loss_coeff=1.0,
                 intrinsic_head=True):

        # Compute the importance weights
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)

        # Compute the KL divergence between the current policy and the previous policy
        action_kl = curr_action_mask *  prev_dist.kl(curr_action_dist)
        self.mean_kl = tf.reduce_mean(action_kl)

        # Compute the entropy of the current policy
        curr_entropy = curr_action_mask * curr_action_dist.entropy()
        self.mean_entropy = tf.reduce_mean(curr_entropy)

        # Compute PPO's clipped surrogate loss
        surrogate_loss = curr_action_mask * tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = tf.reduce_mean(-surrogate_loss)

        # Compute the value function loss
        vf_loss1 = tf.square(value_fn - value_targets)
        vf_clipped = vf_preds + tf.clip_by_value(
            value_fn - vf_preds, -vf_clip_param, vf_clip_param)
        vf_loss2 = tf.square(vf_clipped - value_targets)
        vf_loss = curr_action_mask * tf.maximum(vf_loss1, vf_loss2)
        self.mean_vf_loss = tf.reduce_mean(vf_loss)

        # Compute the intrinsic value function loss
        if intrinsic_head:
            intrinsic_vf_loss1 = tf.square(intrinsic_value_fn - intrinsic_value_targets)
            intrinsic_vf_clipped = intrinsic_vf_preds + tf.clip_by_value(
                intrinsic_value_fn - intrinsic_vf_preds, -vf_clip_param, vf_clip_param)
            intrinsic_vf_loss2 = tf.square(intrinsic_vf_clipped - intrinsic_value_targets)
            intrinsic_vf_loss = curr_action_mask * tf.maximum(intrinsic_vf_loss1, intrinsic_vf_loss2)
            vf_loss += intrinsic_vf_loss
            self.mean_intrinsic_vf_loss = tf.reduce_mean(intrinsic_vf_loss)

        # Compute average policy cloning loss - has to be averaged separately as we have a different batch size
        self.mean_average_policy_loss = -tf.reduce_mean((1.0 - curr_action_mask) * average_action_dist.logp(actions))

        # Combine and return losses
        loss = tf.reduce_mean(
            -surrogate_loss + cur_kl_coeff * action_kl +
            vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy) + self.mean_average_policy_loss

        self.loss = loss


def average_ppo_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    value_function = model.value_function()
    intrinsic_value_function = model.intrinsic_value_function()
    average_logits = model.average_policy_logits()

    action_dist = dist_class(logits, model)
    average_action_dist = dist_class(average_logits, model)

    prev_action_dist = dist_class(train_batch[BEHAVIOUR_LOGITS], model)

    policy.loss_obj = AveragePPOLoss(
        action_dist,
        prev_action_dist,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[INTRINSIC_VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        train_batch[INTRINSIC_VF_PREDS],
        value_function,
        intrinsic_value_function,
        train_batch[CURRENT_ACTION_MASK],
        average_action_dist,
        policy.kl_coeff,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        intrinsic_head=policy.config["intrinsic_head"],
    )

    return policy.loss_obj.loss + model.intrinsic_loss(train_batch)


def loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "intrinsic_vf_loss": policy.loss_obj.mean_intrinsic_vf_loss,
        "average_policy_loss": policy.loss_obj.mean_average_policy_loss,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def intrinsic_loss_stats(policy, batch, grad):
    return policy.model.metrics()


def policy_fetches(policy):
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        INTRINSIC_VF_PREDS: policy.model.intrinsic_value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
    }


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):  # How do we make sure this gets passed in?
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

    # Add action mask
    batch[CURRENT_ACTION_MASK] = np.ones(batch.count)

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


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def make_model(policy, obs_space, action_space, config):
    exploration = not config.get("in_evaluation", False)

    model_config = config.get("model", {}).copy()
    model_config["intrinsic_gamma"] = config.get("gamma", 0.95)
    model_config["vf_share_layers"] = config.get("vf_share_layers", False)
    model_config["num_agents"] = config.get("num_agents", 1)

    return NashModel(obs_space, action_space, action_space.n, model_config, "nash_model", exploration)


DeepNashPolicy = build_tf_policy(
    name="DeepNashPolicy_V1",
    loss_fn=average_ppo_loss,
    stats_fn=loss_stats,
    grad_stats_fn=intrinsic_loss_stats,
    extra_action_fetches_fn=policy_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_loss_init=setup_mixins,
    make_model=make_model,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])
