import logging
import numpy as np
import scipy.signal

import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

from algorithms.agents.intrinsic import INTRINSIC_REWARD
from algorithms.agents.deep_nash_v1.deep_nash_model_v1 import NashModel

logger = logging.getLogger(__name__)


POLICY = "policy"
POLICY_LOGITS = "policy_logits"
Q_VALUES = "q_values"
Q_TARGETS = "q_targets"
IMPORTANCE_WEIGHTS = "importance_weights"


class DeepNashLoss:
    def __init__(self,
                 q_values,
                 q_targets,
                 importance_weights,
                 actions,
                 logits):

        # Compute Q-loss
        self.mean_q_value = tf.reduce_mean(q_values)
        self.mean_q_target = tf.reduce_mean(q_targets)

        advantages = tf.reduce_max(q_values, axis=-1) - tf.reduce_sum(q_values * actions, axis=-1)
        self.mean_advantage = tf.reduce_mean(advantages)

        errors = q_targets - tf.reduce_sum(q_values * actions, axis=-1)
        q_loss = tf.math.maximum(tf.minimum(0.5 * tf.square(errors), 1.0), tf.abs(errors) - 0.5)  # Huber loss
        self.q_loss = tf.reduce_mean(q_loss / importance_weights)

        # Compute policy loss
        log_partition = tf.log(tf.reduce_sum(tf.exp(logits), axis=-1))
        log_likelihood = tf.reduce_sum(logits * actions, axis=-1) - log_partition
        self.policy_loss = -tf.reduce_mean(log_likelihood)

        self.loss = self.q_loss + self.policy_loss


def deep_nash_loss(policy, model, dist_class, train_batch):
    actions = tf.one_hot(train_batch[SampleBatch.ACTIONS], policy.action_space.n, dtype=tf.float32)
    exploration_logits, state = model.from_batch(train_batch)
    q_values = model.q_values()
    logits = model.average_policy_logits()

    policy.loss_obj = DeepNashLoss(
        q_values,
        train_batch[Q_TARGETS],
        train_batch[IMPORTANCE_WEIGHTS],
        actions,
        logits)

    return policy.loss_obj.loss + model.intrinsic_loss(train_batch)


# This is where we compute the intrinsic rewards and Q-function targets
def postprocess_q_targets(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):

    # Add intrinsic reward signal to reward values
    if not (policy.config["in_evaluation"] and policy.loss_initialized()):
        if tf.executing_eagerly():
            sample_batch[INTRINSIC_REWARD] = policy.model.intrinsic_reward(sample_batch)
        else:
            with policy.get_session().as_default():
                sample_batch[INTRINSIC_REWARD] = policy.model.intrinsic_reward(sample_batch)
    
    # Compute importance weighted Q-targets
    actions = np.eye(policy.action_space.n)[sample_batch[SampleBatch.ACTIONS]]
    weights = np.clip(np.sum(actions * sample_batch[POLICY], axis=-1), 0.05, 1.0)

    q_targets = (1.0 - sample_batch[SampleBatch.DONES].astype(float)) * np.sum(actions * sample_batch[Q_VALUES], axis=-1)
    q_targets = scipy.signal.lfilter([0, policy.config["gamma"]], [1], q_targets[::-1], axis=0)[::-1]
    q_targets += sample_batch[SampleBatch.REWARDS]

    if INTRINSIC_REWARD in sample_batch:
        q_targets += sample_batch[INTRINSIC_REWARD]

    batch = { 
        Q_TARGETS: q_targets,
        IMPORTANCE_WEIGHTS: weights,
    }

    for key, value in sample_batch.items():
        batch[key] = sample_batch[key]

    return SampleBatch(batch)


def policy_q_fetches(policy):
    return {
        Q_VALUES: policy.model.q_values(),
        POLICY: policy.model.average_policy(),
        POLICY_LOGITS: policy.model.average_policy_logits(),
    }


def make_model(policy, obs_space, action_space, config):
    exploration = not config.get("in_evaluation", False)
    model_config = config.get("model", {}).copy()
    model_config["intrinsic_gamma"] = config.get("gamma", 0.95)
    model_config["share_layers"] = config.get("share_layers", False)
    model_config["num_agents"] = 1  # Override any specified value

    return NashModel(obs_space, action_space, action_space.n, model_config, "nash_model", exploration)


def policy_loss_stats(policy, train_batch):
    return {
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.policy_loss,
        "q_loss": policy.loss_obj.q_loss,
        "mean_q_value": policy.loss_obj.mean_q_value,
        "mean_q_target": policy.loss_obj.mean_q_target,
        "mean_advantage": policy.loss_obj.mean_advantage,
    }


# This is needed due to an apparent bug in how RLLib 0.8.3 averages training losses
def intrinsic_loss_stats(policy, batch, grad):
    return policy.model.metrics()


class LearningRateMixin:

    def set_learning_rate(self, learning_rate):
        self.model.learning_rate.load(learning_rate, session=self.get_session())


DeepNashPolicy = build_tf_policy(
    name="DeepNashPolicy_V1",
    loss_fn=deep_nash_loss,
    stats_fn=policy_loss_stats,
    grad_stats_fn=intrinsic_loss_stats,
    extra_action_fetches_fn=policy_q_fetches,
    postprocess_fn=postprocess_q_targets,
    make_model=make_model,
    mixins=[
        LearningRateMixin
    ])
