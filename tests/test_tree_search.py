"""Unit tests for Latent Monte Carlo Tree Search (MCTS) planner."""

import numpy as np
import pytest
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.tree_search import LatentMCTSPlanner


def test_latent_mcts_planning():
    """Verify that MCTS explores tree and produces valid discrete action selection."""
    obs_dim = 4
    act_dim = 1
    action_spec = ActionSpec(shape=(1,), minimum=np.array([-1.0], dtype=np.float32), maximum=np.array([1.0], dtype=np.float32))

    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16)

    planner = LatentMCTSPlanner(
        world_model=wm,
        action_spec=action_spec,
        num_simulations=15,
        c_puct=1.414,
    )

    h0 = torch.zeros(1, 16)
    z0 = torch.zeros(1, 4)
    state = RSSMState(deter=h0, stoch=z0, mean=z0, std=torch.ones_like(z0))

    res = planner.plan(state)
    assert res.best_action.shape == (1,)
    assert res.total_simulations == 15
    assert len(res.visit_distribution) == planner.num_actions
    assert np.isclose(np.sum(res.visit_distribution), 1.0)
    assert np.all(res.best_action >= -1.0) and np.all(res.best_action <= 1.0)
