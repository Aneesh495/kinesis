"""Unit tests for Spectral Normalized linear layers."""

import pytest
import torch

from kinesis.models.spectral_norm import SpectralLinear, SpectralMLP


def test_spectral_linear_norm_bounding():
    """Verify that spectral norm scaling caps weight matrix 2-norm."""
    layer = SpectralLinear(in_features=10, out_features=10, max_spectral_norm=1.0)
    layer.train()

    # Artificially blow up layer weight to 10.0
    layer.weight.data *= 10.0

    x = torch.randn(4, 10)
    # Forward passes update power iteration singular vectors
    for _ in range(5):
        out = layer(x)

    # Compute actual singular values of weight via SVD
    with torch.no_grad():
        sigma_est = layer.compute_spectral_norm(update_buffers=True)
        # Scale factor should be greater than 1
        assert sigma_est.item() > 1.0

        # Effective normalized weight matrix
        scale = max(1.0, sigma_est.item() / layer.max_norm)
        eff_w = layer.weight / scale
        s_true = torch.linalg.svdvals(eff_w)
        # Largest singular value of effective weight should be approximately bounded by max_norm
        assert s_true[0].item() <= 1.05


def test_spectral_mlp_backward():
    """Verify gradient backpropagation through SpectralMLP."""
    mlp = SpectralMLP(in_dim=4, out_dim=2, hidden_dim=16, num_layers=2)
    x = torch.randn(3, 4, requires_grad=True)

    out = mlp(x)
    assert out.shape == (3, 2)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
