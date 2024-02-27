"""Unit tests for Vector Quantized RSSM."""

import pytest
import torch

from kinesis.models.vq_vae import VectorQuantizer, VectorQuantizedRSSM


def test_vector_quantizer_straight_through():
    """Verify codebook assignment, perplexity, and gradient backpropagation."""
    quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8, commitment_cost=0.25)
    inputs = torch.randn(4, 8, requires_grad=True)

    out = quantizer(inputs)
    assert out.quantized.shape == (4, 8)
    assert out.indices.shape == (4,)
    assert out.loss.item() >= 0.0
    assert out.perplexity.item() >= 1.0

    # Test backward pass through straight-through estimator
    loss = out.quantized.sum() + out.loss
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))


def test_vq_rssm_observe_and_imagine():
    """Verify VQ-RSSM observe and imagine forward paths."""
    model = VectorQuantizedRSSM(
        obs_dim=5,
        action_dim=1,
        embed_dim=16,
        deter_dim=32,
        num_embeddings=32,
        embedding_dim=8,
        hidden_dim=32,
    )

    B = 2
    device = torch.device("cpu")
    h0, z0 = model.initial_state(B, device)

    obs = torch.randn(B, 5)
    act = torch.randn(B, 1)

    # Observe step
    h1, post_vq, prior_vq = model.observe_step(h0, z0, act, obs)
    assert h1.shape == (B, 32)
    assert post_vq.quantized.shape == (B, 8)
    assert prior_vq.quantized.shape == (B, 8)

    # Imagine step
    h_im, prior_im = model.imagine_step(h1, post_vq.quantized, act)
    assert h_im.shape == (B, 32)
    assert prior_im.quantized.shape == (B, 8)

    # Prediction heads
    feat = torch.cat([h_im, prior_im.quantized], dim=-1)
    rec_obs = model.obs_head(feat)
    pred_rew = model.reward_head(feat)
    assert rec_obs.shape == (B, 5)
    assert pred_rew.shape == (B, 1)
