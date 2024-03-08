"""Unit tests for CMA-ES latent trajectory planner."""

import numpy as np
import pytest
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.world_model import WorldModel
from kinesis.planning.cma_es import CMAESPlanner


def test_cma_es_planner_optimization():
    """Verify that CMA-ES optimizes latent trajectories within bounds."""
    obs_dim = 4
    act_dim = 1
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))

    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16)
    planner = CMAESPlanner(
        world_model=wm,
        action_spec=action_spec,
        horizon=6,
        population_size=12,
        num_elites=4,
        max_iterations=3,
        initial_std=0.5,
    )

    h0 = torch.zeros(1, 16)
    z0 = torch.zeros(1, 4)

    res = planner.plan(h0, z0)
    assert res.best_action_sequence.shape == (6, 1)
    assert np.all(res.best_action_sequence >= -1.0)
    assert np.all(res.best_action_sequence <= 1.0)
    assert np.isfinite(res.best_return)
    assert len(res.mean_history) == 3
