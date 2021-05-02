import numpy as np
import scipy.signal
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing

from algorithms.curiosity import INTRINSIC_REWARD

INTRINSIC_VALUE_TARGETS = "intrinsic_value_targets"
INTRINSIC_VF_PREDS = "intrinsic_vf_preds"


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_advantages_intrinsic(rollout,
                       last_r,
                       last_intrinsic_r,
                       gamma=0.9,
                       intrinsic_gamma=0.9,
                       lambda_=1.0,
                       intrinsic_lambda_=1.0):
    """
    Given a rollout, compute its value targets and the advantage. Assumes we are using separate
    value function heads for the extrinsic and intrinsic rewards
    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor
        intrinsic_gamma (float): Discount factor
        lambda_ (float): Parameter for GAE
        intrinsic_lambda_ (float): Parameter for intrinsic GAE
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    # Extrinsic value predictions and targets
    vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([last_r])])
    delta_t = (traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
    advantages = discount(delta_t, gamma * lambda_)

    traj[Postprocessing.VALUE_TARGETS] = (
        advantages + traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)

    # Intrinsic value predictions
    intrinsic_vpred_t = np.concatenate([rollout[INTRINSIC_VF_PREDS], np.array([last_intrinsic_r])])
    intrinsic_delta_t = (traj[INTRINSIC_REWARD] + intrinsic_gamma * intrinsic_vpred_t[1:] - intrinsic_vpred_t[:-1])
    intrinsic_advantages = discount(intrinsic_delta_t, intrinsic_gamma * intrinsic_lambda_)

    traj[INTRINSIC_VALUE_TARGETS] = (
        intrinsic_advantages + traj[INTRINSIC_VF_PREDS]).copy().astype(np.float32)

    traj[Postprocessing.ADVANTAGES] = (advantages + intrinsic_advantages).copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"

    return SampleBatch(traj)
