"""RealNVP Normalizing Flow prior density head for expressive multimodal latent transitions."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class FlowOutput:
    """Output from normalizing flow transformation."""
    transformed: torch.Tensor
    log_abs_det_jacobian: torch.Tensor
    log_prob: torch.Tensor


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer splitting input into two halves and applying scale and translation."""

    def __init__(self, dim: int, hidden_dim: int = 64, parity: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.rest_dim = dim - self.split_dim
        self.parity = parity

        # Networks producing scale s and translation t
        in_d = self.rest_dim if parity else self.split_dim
        out_d = 2 * (self.split_dim if parity else self.rest_dim)

        self.net = nn.Sequential(
            nn.Linear(in_d, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_d),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: z -> y with log determinant of Jacobian."""
        if self.parity:
            x1, x2 = x[:, :self.rest_dim], x[:, self.rest_dim:]
            st = self.net(x1)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.tanh(s)  # Stability clamp
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=-1)
            log_det = s.sum(dim=-1)
        else:
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            st = self.net(x1)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.tanh(s)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=-1)
            log_det = s.sum(dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass: y -> z with negative log determinant."""
        if self.parity:
            y1, y2 = y[:, :self.rest_dim], y[:, self.rest_dim:]
            st = self.net(y1)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.tanh(s)
            x2 = (y2 - t) * torch.exp(-s)
            x = torch.cat([y1, x2], dim=-1)
            log_det = -s.sum(dim=-1)
        else:
            y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]
            st = self.net(y1)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.tanh(s)
            x2 = (y2 - t) * torch.exp(-s)
            x = torch.cat([y1, x2], dim=-1)
            log_det = -s.sum(dim=-1)

        return x, log_det


class RealNVPFlow(nn.Module):
    """Composite RealNVP normalizing flow chaining alternating affine coupling layers."""

    def __init__(self, latent_dim: int, num_layers: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(latent_dim, hidden_dim, parity=(i % 2 == 1))
            for i in range(num_layers)
        ])

    def forward(self, z: torch.Tensor) -> FlowOutput:
        """Transform base distribution sample z ~ N(0, I) to complex latent space y."""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        curr = z
        for layer in self.layers:
            curr, log_det = layer(curr)
            log_det_sum = log_det_sum + log_det

        # Base standard normal log prob: -0.5 * (d * log(2pi) + ||z||^2)
        d = self.latent_dim
        base_log_prob = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device)) + torch.sum(z**2, dim=-1))
        # Change of variables: log p(y) = log p(z) - log |det(df/dz)|
        log_prob = base_log_prob - log_det_sum

        return FlowOutput(
            transformed=curr,
            log_abs_det_jacobian=log_det_sum,
            log_prob=log_prob,
        )

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert complex latent state back to base Gaussian coordinates."""
        log_det_sum = torch.zeros(y.shape[0], device=y.device)
        curr = y
        for layer in reversed(self.layers):
            curr, log_det = layer.inverse(curr)
            log_det_sum = log_det_sum + log_det

        return curr, log_det_sum
