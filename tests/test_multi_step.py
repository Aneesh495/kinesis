"""Unit tests for Multi-step Jumpy Rollout Predictor."""

import pytest
import torch

from kinesis.models.multi_step import MultiStepJumpyPredictor


def test_multi_step_jumpy_prediction():
    """Verify that multi-step predictor maps action sequences to future latent states."""
    predictor = MultiStepJumpyPredictor(
        latent_dim=16,
        action_dim=2,
        action_embedding_dim=8,
        hidden_dim=32,
        max_horizon=10,
    )

    B = 4
    k = 7
    h0 = torch.randn(B, 16)
    actions = torch.randn(B, k, 2)
    h_target = torch.randn(B, 16)

    out = predictor(h0, actions, h_target)
    assert out.predicted_latent.shape == (B, 16)
    assert out.horizon_steps == k
    assert out.multi_step_loss.item() > 0.0

    # Test backward pass
    out.multi_step_loss.backward()
    for param in predictor.parameters():
        if param.requires_grad:
            assert param.grad is not None
