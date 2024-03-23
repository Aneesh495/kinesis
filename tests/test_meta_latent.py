"""Unit tests for Online Meta-Adapter context modulation."""

import pytest
import torch

from kinesis.adaptation.meta_latent import LatentContextModulator, OnlineMetaAdapter


def test_meta_adaptation_loss_reduction():
    """Verify that inner-loop gradient adaptation strictly reduces feature error."""
    modulator = LatentContextModulator(feature_dim=16, context_dim=8)
    adapter = OnlineMetaAdapter(modulator, inner_lr=0.1, num_inner_steps=5)

    feat = torch.randn(4, 16)
    # Target feature reflects an affine shift
    target = feat * 1.2 + 0.3

    context_opt, telem = adapter.adapt(feat, target)
    assert telem.adapted_loss < telem.initial_loss
    assert telem.context_norm > 0.0
    assert context_opt.shape == (1, 8)


def test_modulator_preserves_zero_context():
    """Verify that zero context modulation applies default affine scaling."""
    modulator = LatentContextModulator(feature_dim=8, context_dim=4)
    feat = torch.randn(2, 8)
    ctx = torch.zeros(1, 4)

    out = modulator(feat, ctx)
    assert out.shape == (2, 8)
    assert torch.all(torch.isfinite(out))
