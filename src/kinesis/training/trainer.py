"""World model training coordinator with batch updates and gradient clipping."""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from kinesis.models.losses import WorldModelLossResult
from kinesis.models.world_model import WorldModel
from kinesis.replay.buffer import BoundedReplayBuffer


@dataclass(frozen=True)
class TrainingStepMetrics:
    """Detailed loss metrics and diagnostic signals for a training step."""
    step: int
    total_loss: float
    recon_loss: float
    kl_loss: float
    reward_loss: float
    discount_loss: float
    constraint_loss: float
    latent_usage_ratio: float
    grad_norm: float


class WorldModelTrainer:
    """Manages optimizer steps, replay sampling, and parameter updates for WorldModel."""

    def __init__(
        self,
        model: WorldModel,
        lr: float = 3e-4,
        grad_clip: float = 100.0,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)

        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.step_count = 0

    def train_batch(
        self,
        replay_buffer: BoundedReplayBuffer,
        batch_size: int = 16,
        sequence_length: int = 16,
    ) -> TrainingStepMetrics:
        """Sample sequence batch from replay and perform one gradient update step."""
        batch = replay_buffer.sample_sequences(batch_size=batch_size, sequence_length=sequence_length)

        # Move tensors to compute device
        obs_tensor = torch.cat(list(batch["observations"].values()), dim=-1).to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        discounts = batch["discounts"].to(self.device)
        costs = batch["constraint_costs"].to(self.device)
        is_first = batch["is_first"].to(self.device)

        self.optimizer.zero_grad()
        loss_res: WorldModelLossResult = self.model.compute_losses(
            observations=obs_tensor,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            constraint_costs=costs,
            is_first=is_first,
        )

        loss_res.total_loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip))
        self.optimizer.step()
        self.step_count += 1

        return TrainingStepMetrics(
            step=self.step_count,
            total_loss=float(loss_res.total_loss.item()),
            recon_loss=float(loss_res.recon_loss.item()),
            kl_loss=float(loss_res.kl_loss.item()),
            reward_loss=float(loss_res.reward_loss.item()),
            discount_loss=float(loss_res.discount_loss.item()),
            constraint_loss=float(loss_res.constraint_loss.item()),
            latent_usage_ratio=loss_res.latent_usage_ratio,
            grad_norm=grad_norm,
        )
