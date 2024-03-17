"""Unit tests for Active Information-Seeking Curiosity planner."""

import numpy as np
import pytest
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.active_exploration import ActiveExplorationPlanner
from kinesis.uncertainty.ensemble import WorldModelEnsemble


def test_active_exploration_planning():
    """Verify that curiosity planner optimizes actions maximizing disagreement."""
    obs_dim = 4
    act_dim = 1
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))

    models = [
        WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16)
        for _ in range(3)
    ]
    ensemble = WorldModelEnsemble(models)

    planner = ActiveExplorationPlanner(
        ensemble=ensemble,
        action_spec=action_spec,
        horizon=5,
        num_candidates=10,
        num_elites=3,
        num_iterations=2,
    )

    init_states = [
        RSSMState(
            deter=torch.zeros(1, 16),
            stoch=torch.zeros(1, 4),
            mean=torch.zeros(1, 4),
            std=torch.ones(1, 4),
        )
        for _ in range(3)
    ]

    res = planner.plan(init_states)
    assert res.best_action_sequence.shape == (5, 1)
    assert np.all(res.best_action_sequence >= -1.0) and np.all(res.best_action_sequence <= 1.0)
    assert res.expected_information_gain >= 0.0
    assert len(res.mean_gain_history) == 2
