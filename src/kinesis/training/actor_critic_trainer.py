"""Actor-critic policy training coordinator operating on imagined latent trajectories."""

from dataclasses import dataclass
from typing import Any
import torch

from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.policies.latent_actor_critic import ImaginedActorCritic
from kinesis.replay.buffer import BoundedReplayBuffer


@dataclass(frozen=True)
class PolicyTrainingMetrics:
    step: int
    actor_loss: float
    critic_loss: float
    mean_imagined_reward: float


class PolicyTrainer:
    """Manages policy and value function updates on imagined rollout batches."""

    def __init__(
        self,
        policy: ImaginedActorCritic,
        world_model: WorldModel,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.world_model = world_model
        self.device = device if device is not None else torch.device("cpu")
        self.policy.to(self.device)
        self.world_model.to(self.device)
        self.step_count = 0

    def train_step(
        self,
        replay_buffer: BoundedReplayBuffer,
        batch_size: int = 16,
        prefix_length: int = 8,
    ) -> PolicyTrainingMetrics:
        """Sample sequence prefix from replay, unroll posterior to get realistic init_states, and train policy."""
        batch = replay_buffer.sample_sequences(batch_size=batch_size, sequence_length=prefix_length)

        obs_tensor = torch.cat(list(batch["observations"].values()), dim=-1).to(self.device)
        actions = batch["actions"].to(self.device)
        is_first = batch["is_first"].to(self.device)

        # Unroll belief states on real replay prefix
        with torch.no_grad():
            embeds = self.world_model.encoder(obs_tensor)
            _, post_states = self.world_model.rssm.observe_sequence(embeds, actions, is_first)
            # Use final posterior belief state as initial state for imagination
            final_post = post_states[-1]

        # Optimize actor and critic on imagined rollout
        metrics = self.policy.train_on_imagined_trajectories(final_post)
        self.step_count += 1

        return PolicyTrainingMetrics(
            step=self.step_count,
            actor_loss=metrics["actor_loss"],
            critic_loss=metrics["critic_loss"],
            mean_imagined_reward=metrics["mean_imagined_reward"],
        )
