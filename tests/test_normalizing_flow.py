"""Unit tests for RealNVP Normalizing Flow."""

import pytest
import torch

from kinesis.models.normalizing_flow import RealNVPFlow


def test_real_nvp_exact_inversion():
    """Verify that RealNVP flow is strictly invertible: f^(-1)(f(z)) == z."""
    flow = RealNVPFlow(latent_dim=6, num_layers=4, hidden_dim=32)
    flow.eval()

    z = torch.randn(5, 6)
    out = flow(z)
    y = out.transformed

    z_rec, log_det_inv = flow.inverse(y)

    assert torch.allclose(z, z_rec, atol=1e-5)
    # Forward and inverse log determinants should cancel out
    assert torch.allclose(out.log_abs_det_jacobian, -log_det_inv, atol=1e-5)


def test_real_nvp_log_prob_computation():
    """Verify that log-probability computation produces finite values and supports backprop."""
    flow = RealNVPFlow(latent_dim=4, num_layers=2, hidden_dim=16)
    z = torch.randn(3, 4, requires_grad=True)

    out = flow(z)
    assert out.log_prob.shape == (3,)
    assert torch.all(torch.isfinite(out.log_prob))

    # Loss optimization
    loss = -out.log_prob.mean()
    loss.backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))
