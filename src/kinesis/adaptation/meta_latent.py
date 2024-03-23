"""Online fast meta-adaptation of latent transition dynamics via context modulation."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MetaAdaptationResult:
    """Telemetry from inner-loop online meta-adaptation."""
    initial_loss: float
    adapted_loss: float
    context_norm: float
    adaptation_steps: int


class LatentContextModulator(nn.Module):
    """Context modulation layer injecting adaptive latent shifts into transition embeddings."""

    def __init__(self, feature_dim: int, context_dim: int = 16) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim

        # Context adapter produces scale and shift vectors: gamma(psi), beta(psi)
        self.adapter = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Modulate features using FiLM-style scale and shift: h * (1 + gamma) + beta."""
        gamma_beta = self.adapter(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return features * (1.0 + gamma) + beta


class OnlineMetaAdapter:
    """Inner-loop gradient optimizer adapting context modulation on recent transitions."""

    def __init__(
        self,
        modulator: LatentContextModulator,
        inner_lr: float = 0.05,
        num_inner_steps: int = 3,
    ) -> None:
        self.modulator = modulator
        self.inner_lr = inner_lr
        self.num_steps = num_inner_steps

    def adapt(
        self,
        features: torch.Tensor,       # [B, feature_dim]
        target_features: torch.Tensor, # [B, feature_dim]
        initial_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MetaAdaptationResult]:
        """Run few-step inner loop gradient descent on context vector."""
        device = features.device
        if initial_context is not None:
            context = initial_context.clone().detach().requires_grad_(True)
        else:
            context = torch.zeros(1, self.modulator.context_dim, device=device, requires_grad=True)

        optimizer = torch.optim.SGD([context], lr=self.inner_lr)

        with torch.enable_grad():
            init_out = self.modulator(features, context)
            init_loss = float(F.mse_loss(init_out, target_features).item())

            for _ in range(self.num_steps):
                optimizer.zero_grad()
                adapted_feat = self.modulator(features, context)
                loss = F.mse_loss(adapted_feat, target_features)
                loss.backward()
                optimizer.step()

            final_out = self.modulator(features, context)
            final_loss = float(F.mse_loss(final_out, target_features).item())

        telemetry = MetaAdaptationResult(
            initial_loss=init_loss,
            adapted_loss=final_loss,
            context_norm=float(torch.norm(context).item()),
            adaptation_steps=self.num_steps,
        )

        return context.detach(), telemetry
