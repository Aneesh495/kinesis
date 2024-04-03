"""Unit tests for Variational Information Bottleneck."""

import pytest
import torch

from kinesis.models.bottleneck import VariationalInformationBottleneck


def test_vib_kl_divergence_and_differentiability():
    """Verify that VIB computes positive KL divergence and backpropagates gradients."""
    vib = VariationalInformationBottleneck(in_dim=8, latent_dim=4, target_kl=1.0)
    vib.train()

    x = torch.randn(5, 8, requires_grad=True)
    task_loss = torch.tensor(2.0, requires_grad=True)

    out = vib(x, task_loss)
    assert out.latent.shape == (5, 4)
    assert out.kl_divergence.item() >= 0.0
    assert out.total_loss.requires_grad is True

    out.total_loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_vib_deterministic_eval_mode():
    """Verify that eval mode returns the deterministic mean without noise injection."""
    vib = VariationalInformationBottleneck(in_dim=6, latent_dim=3)
    vib.eval()

    x = torch.randn(2, 6)
    out1 = vib(x, torch.tensor(0.0))
    out2 = vib(x, torch.tensor(0.0))

    # Latents should be completely identical in eval mode
    assert torch.allclose(out1.latent, out2.latent)
    assert torch.allclose(out1.latent, out1.mean)
