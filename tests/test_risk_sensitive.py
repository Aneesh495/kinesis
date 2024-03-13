"""Unit tests for Risk-Sensitive CVaR and Entropic latent planner."""

import numpy as np
import pytest
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.risk_sensitive import RiskMeasure, RiskSensitivePlanner


def test_risk_metrics_monotonicity():
    """Verify that CVaR is strictly less than or equal to mean return."""
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))
    wm = WorldModel(obs_dim=3, action_dim=1, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16)

    planner = RiskSensitivePlanner(
        world_model=wm,
        action_spec=action_spec,
        risk_measure=RiskMeasure.CVAR,
        alpha_cvar=0.2,
    )

    # 3 candidates with varying returns across 10 rollouts
    returns = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [-10.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    ])

    scores, cvars, mean_rets = planner.evaluate_risk(returns)
    for cvar, mean_val in zip(cvars, mean_rets):
        assert cvar <= mean_val + 1e-6


def test_risk_sensitive_plan_execution():
    """Verify that RiskSensitivePlanner produces bounded action sequence."""
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))
    wm = WorldModel(obs_dim=3, action_dim=1, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16)

    planner = RiskSensitivePlanner(
        world_model=wm,
        action_spec=action_spec,
        horizon=4,
        num_candidates=6,
        rollouts_per_candidate=2,
        num_iterations=2,
    )

    h0 = torch.zeros(1, 16)
    z0 = torch.zeros(1, 4)
    state = RSSMState(deter=h0, stoch=z0, mean=z0, std=torch.ones_like(z0))

    res = planner.plan(state)
    assert res.best_action_sequence.shape == (4, 1)
    assert np.all(res.best_action_sequence >= -1.0) and np.all(res.best_action_sequence <= 1.0)
    assert res.cvar_value <= res.mean_return + 1e-6
