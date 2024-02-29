"""Unit tests for Causal Transformer World Model."""

import pytest
import torch

from kinesis.models.attention import CausalTransformerWorldModel


def test_transformer_world_model_causality():
    """Verify strictly causal information flow: modifying future tokens does not affect past outputs."""
    model = CausalTransformerWorldModel(
        obs_dim=4,
        action_dim=2,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
    )
    model.eval()

    obs = torch.randn(2, 10, 4)
    act = torch.randn(2, 10, 2)

    with torch.no_grad():
        out1 = model(obs, act)

    # Mutate only future tokens at t >= 5
    obs_mutated = obs.clone()
    act_mutated = act.clone()
    obs_mutated[:, 5:] = torch.randn(2, 5, 4) * 10.0
    act_mutated[:, 5:] = torch.randn(2, 5, 2) * 10.0

    with torch.no_grad():
        out2 = model(obs_mutated, act_mutated)

    # Outputs at timesteps 0 through 4 must be identical within float precision
    assert torch.allclose(out1.pred_observations[:, :5], out2.pred_observations[:, :5], atol=1e-5)
    assert torch.allclose(out1.pred_rewards[:, :5], out2.pred_rewards[:, :5], atol=1e-5)
    # Outputs at t >= 5 should diverge due to mutation
    assert not torch.allclose(out1.pred_observations[:, 5:], out2.pred_observations[:, 5:])


def test_transformer_world_model_backward():
    """Verify gradient computation through full sequence."""
    model = CausalTransformerWorldModel(
        obs_dim=3,
        action_dim=1,
        d_model=16,
        num_heads=2,
        num_layers=2,
    )
    obs = torch.randn(2, 8, 3, requires_grad=True)
    act = torch.randn(2, 8, 1, requires_grad=True)

    out = model(obs, act)
    loss = out.pred_observations.sum() + out.pred_rewards.sum()
    loss.backward()

    assert obs.grad is not None
    assert act.grad is not None
    assert torch.all(torch.isfinite(obs.grad))
