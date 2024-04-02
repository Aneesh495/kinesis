"""Spectral normalized dense layers enforcing strict Lipschitz continuity in latent dynamics."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLinear(nn.Module):
    """Linear layer with power-iteration spectral normalization bounding matrix 2-norm."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        max_spectral_norm: float = 1.0,
        power_iterations: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_norm = float(max_spectral_norm)
        self.power_iterations = power_iterations

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Persistent buffers for left and right singular vector estimates
        self.register_buffer("u_vec", torch.randn(out_features))
        self.register_buffer("v_vec", torch.randn(in_features))
        self._normalize_buffers()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _normalize_buffers(self) -> None:
        """Ensure u and v vectors have unit Euclidean norm."""
        self.u_vec.data = F.normalize(self.u_vec.data, dim=0, eps=1e-12)
        self.v_vec.data = F.normalize(self.v_vec.data, dim=0, eps=1e-12)

    def compute_spectral_norm(self, update_buffers: bool = True) -> torch.Tensor:
        """Estimate largest singular value via power iteration."""
        u = self.u_vec
        v = self.v_vec
        w = self.weight

        if update_buffers and self.training:
            with torch.no_grad():
                for _ in range(self.power_iterations):
                    # v = W^T * u / ||W^T * u||
                    v = F.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
                    # u = W * v / ||W * v||
                    u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
                self.u_vec.copy_(u)
                self.v_vec.copy_(v)

        # sigma = u^T * W * v
        sigma = torch.dot(u, torch.mv(w, v))
        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalized weights: W_norm = W / max(1, sigma / max_norm)."""
        sigma = self.compute_spectral_norm(update_buffers=self.training)
        # Scale down if sigma exceeds maximum allowed Lipschitz constant
        scale = torch.clamp(sigma / self.max_norm, min=1.0)
        norm_weight = self.weight / scale
        return F.linear(x, norm_weight, self.bias)


class SpectralMLP(nn.Module):
    """Multi-layer perceptron with spectral normalization on all linear transformations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_spectral_norm: float = 1.0,
    ) -> None:
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(SpectralLinear(curr_dim, hidden_dim, max_spectral_norm=max_spectral_norm))
            layers.append(nn.ELU())
            curr_dim = hidden_dim

        layers.append(SpectralLinear(curr_dim, out_dim, max_spectral_norm=max_spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
