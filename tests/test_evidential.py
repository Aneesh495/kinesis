"""Unit tests for Deep Evidential Regression."""

import pytest
import torch

from kinesis.uncertainty.evidential import EvidentialRegressionHead, evidential_regression_loss


def test_evidential_head_support_constraints():
    """Verify that evidential head strictly outputs valid NIG support parameters."""
    head = EvidentialRegressionHead(in_features=16, out_features=3)
    x = torch.randn(5, 16)

    out = head(x)
    assert out.gamma.shape == (5, 3)
    assert out.v.shape == (5, 3)
    assert out.alpha.shape == (5, 3)
    assert out.beta.shape == (5, 3)

    # Support constraints
    assert torch.all(out.v > 0.0)
    assert torch.all(out.alpha > 1.0)
    assert torch.all(out.beta > 0.0)
    assert torch.all(out.aleatoric_variance > 0.0)
    assert torch.all(out.epistemic_variance > 0.0)


def test_evidential_loss_optimization():
    """Verify that evidential loss decreases over gradient steps."""
    torch.manual_seed(42)
    head = EvidentialRegressionHead(in_features=8, out_features=1)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    x = torch.randn(10, 8)
    target = 2.0 * x[:, 0:1] + 0.5

    init_loss = evidential_regression_loss(head(x), target).item()

    for _ in range(20):
        optimizer.zero_grad()
        loss = evidential_regression_loss(head(x), target)
        loss.backward()
        optimizer.step()

    final_loss = evidential_regression_loss(head(x), target).item()
    assert final_loss < init_loss
