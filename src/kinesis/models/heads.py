"""Prediction heads for reward, continuation discount, and constraint costs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    """Predicts expected scalar reward from latent features [h_t, z_t]."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = feature_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Output scalar predicted reward [..., 1]."""
        return self.net(features)


class DiscountHead(nn.Module):
    """Predicts continuation discount factor gamma in [0, 1] (Bernoulli termination)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = feature_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns continuation probability [..., 1]."""
        logits = self.net(features)
        return torch.sigmoid(logits)

    def compute_loss(self, features: torch.Tensor, target_discounts: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        target = (target_discounts > 0.0).float()
        return F.binary_cross_entropy_with_logits(logits, target)


class ConstraintHead(nn.Module):
    """Predicts non-negative constraint violation cost from latent features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = feature_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns predicted cost c >= 0 via softplus."""
        return F.softplus(self.net(features))
