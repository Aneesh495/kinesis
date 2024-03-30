"""Multi-step jumpy rollout predictor predicting multi-horizon latent transitions."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JumpyPredictionOutput:
    """Output from multi-step jumpy predictor."""
    predicted_latent: torch.Tensor  # [B, latent_dim]
    multi_step_loss: torch.Tensor
    horizon_steps: int


class MultiStepJumpyPredictor(nn.Module):
    """Predicts future latent state k steps ahead without intermediate observation decoding."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_embedding_dim: int = 32,
        hidden_dim: int = 64,
        max_horizon: int = 15,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon

        # Action sequence aggregator: compresses action sequence a_{t:t+k} into vector
        self.action_gru = nn.GRU(action_dim, action_embedding_dim, batch_first=True)

        # Transition jump MLP: maps [h_t, action_seq_summary] -> h_{t+k}
        self.jump_net = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        initial_latent: torch.Tensor,       # [B, latent_dim]
        action_sequence: torch.Tensor,      # [B, k, action_dim]
        target_future_latent: torch.Tensor | None = None, # [B, latent_dim]
    ) -> JumpyPredictionOutput:
        """Predict latent state at end of action sequence and compute multi-step loss."""
        B, k, _ = action_sequence.shape
        assert k <= self.max_horizon, f"Horizon {k} exceeds max_horizon {self.max_horizon}"

        # Aggregate action sequence
        _, hn = self.action_gru(action_sequence)
        action_summary = hn.squeeze(0)  # [B, action_embedding_dim]

        # Predict future state
        combined = torch.cat([initial_latent, action_summary], dim=-1)
        pred_future = self.jump_net(combined)

        if target_future_latent is not None:
            loss = F.mse_loss(pred_future, target_future_latent)
        else:
            loss = torch.tensor(0.0, device=initial_latent.device)

        return JumpyPredictionOutput(
            predicted_latent=pred_future,
            multi_step_loss=loss,
            horizon_steps=k,
        )
