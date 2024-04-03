"""Variational Information Bottleneck (VIB) regularizer with adaptive Lagrange multiplier control."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BottleneckOutput:
    """Output from variational information bottleneck layer."""
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    kl_divergence: torch.Tensor
    total_loss: torch.Tensor
    current_beta: float


class VariationalInformationBottleneck(nn.Module):
    """Enforces representation compression by bounding mutual information I(X; Z)."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        target_kl: float = 2.0,  # Target mutual information in nats
        initial_beta: float = 1e-2,
        adaptive_beta: bool = True,
        beta_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.target_kl = float(target_kl)
        self.adaptive = adaptive_beta
        self.beta_lr = beta_lr

        self.fc_mean = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Dual Lagrange multiplier log(beta) for adaptive rate control
        self.log_beta = nn.Parameter(torch.log(torch.tensor(float(initial_beta))), requires_grad=adaptive_beta)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vector using reparameterization trick: z = mean + eps * exp(0.5 * logvar)."""
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, min=-10.0, max=5.0))
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x: torch.Tensor, task_loss: torch.Tensor) -> BottleneckOutput:
        """Encode input and compute VIB objective: TaskLoss + beta * (KL - TargetKL)."""
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mean, logvar)

        # KL divergence to standard normal N(0, I)
        # KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_elementwise = -0.5 * (1.0 + logvar - mean**2 - torch.exp(logvar))
        kl = torch.mean(torch.sum(kl_elementwise, dim=-1))

        current_beta = float(torch.exp(self.log_beta).item())
        vib_penalty = torch.exp(self.log_beta) * (kl - self.target_kl)
        total_loss = task_loss + vib_penalty

        return BottleneckOutput(
            latent=z,
            mean=mean,
            logvar=logvar,
            kl_divergence=kl,
            total_loss=total_loss,
            current_beta=current_beta,
        )
