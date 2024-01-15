"""Stochastic Recurrent State-Space Model (RSSM) with deterministic GRU and diagonal Gaussian latents."""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinesis.contracts.types import LatentShape


@dataclass
class RSSMState:
    """Belief state consisting of deterministic recurrent vector and stochastic latent sample."""
    deter: torch.Tensor  # [..., deter_dim]
    stoch: torch.Tensor  # [..., stoch_dim]
    mean: torch.Tensor   # [..., stoch_dim]
    std: torch.Tensor    # [..., stoch_dim]

    @property
    def features(self) -> torch.Tensor:
        """Concatenated feature vector [h_t, z_t] for decision heads and value estimation."""
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            mean=self.mean.detach(),
            std=self.std.detach(),
        )


class RecurrentStateSpaceModel(nn.Module):
    """Original Stochastic Recurrent State-Space Model:

    Deterministic update:
      h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    Stochastic Prior:
      p(z_t | h_t) = N(mu_prior(h_t), sigma_prior(h_t))

    Stochastic Posterior:
      q(z_t | h_t, e_t) = N(mu_post(h_t, e_t), sigma_post(h_t, e_t))
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        self.max_std = max_std

        self.latent_shape = LatentShape(deterministic_dim=deter_dim, stochastic_dim=stoch_dim)

        # Deterministic recurrent GRU cell
        self.gru_cell = nn.GRUCell(input_size=stoch_dim + action_dim, hidden_size=deter_dim)

        # Prior network: h_t -> [mean, std]
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # Posterior network: [h_t, e_t] -> [mean, std]
        self.post_mlp = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Create zero-initialized belief state."""
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def _parameterize_gaussian(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        # Bounded softplus parameterization for numerical stability
        std = 2.0 * torch.sigmoid(raw_std / 2.0) + self.min_std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        return mean, std

    def _sample_gaussian(
        self, mean: torch.Tensor, std: torch.Tensor, sample_mode: bool = True
    ) -> torch.Tensor:
        if not sample_mode:
            return mean
        eps = torch.randn_like(mean)
        return mean + std * eps

    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        """Compute one step of both prior and observation-conditioned posterior."""
        # Recurrent deterministic step: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        gru_in = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.gru_cell(gru_in, prev_state.deter)

        # Prior distribution p(z_t | h_t)
        prior_stats = self.prior_mlp(deter)
        prior_mean, prior_std = self._parameterize_gaussian(prior_stats)
        prior_stoch = self._sample_gaussian(prior_mean, prior_std, sample_mode=sample)
        prior_state = RSSMState(deter=deter, stoch=prior_stoch, mean=prior_mean, std=prior_std)

        # Posterior distribution q(z_t | h_t, e_t)
        post_in = torch.cat([deter, embed], dim=-1)
        post_stats = self.post_mlp(post_in)
        post_mean, post_std = self._parameterize_gaussian(post_stats)
        post_stoch = self._sample_gaussian(post_mean, post_std, sample_mode=sample)
        post_state = RSSMState(deter=deter, stoch=post_stoch, mean=post_mean, std=post_std)

        return prior_state, post_state

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        sample: bool = True,
    ) -> RSSMState:
        """Step transition dynamics forward strictly using the prior distribution (no observations)."""
        gru_in = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.gru_cell(gru_in, prev_state.deter)

        prior_stats = self.prior_mlp(deter)
        mean, std = self._parameterize_gaussian(prior_stats)
        stoch = self._sample_gaussian(mean, std, sample_mode=sample)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def observe_sequence(
        self,
        embeds: torch.Tensor,  # [B, T, embed_dim]
        actions: torch.Tensor,  # [B, T, action_dim]
        is_first: torch.Tensor,  # [B, T]
        init_state: RSSMState | None = None,
    ) -> tuple[list[RSSMState], list[RSSMState]]:
        """Process sequence of T steps forward in time, resetting state at episode boundaries."""
        B, T, _ = embeds.shape
        device = embeds.device

        state = init_state if init_state is not None else self.initial_state(B, device)

        prior_states: list[RSSMState] = []
        post_states: list[RSSMState] = []

        for t in range(T):
            # Mask state with is_first indicator (1.0 = first step, reset state)
            mask = 1.0 - is_first[:, t].unsqueeze(-1)
            masked_deter = state.deter * mask
            masked_stoch = state.stoch * mask
            masked_prev = RSSMState(
                deter=masked_deter,
                stoch=masked_stoch,
                mean=state.mean * mask,
                std=state.std * mask + (1.0 - mask),
            )

            act_t = actions[:, t]
            emb_t = embeds[:, t]

            prior_s, post_s = self.observe_step(masked_prev, act_t, emb_t)
            prior_states.append(prior_s)
            post_states.append(post_s)
            state = post_s

        return prior_states, post_states

    def imagine_trajectory(
        self,
        init_state: RSSMState,
        action_sequence: torch.Tensor,  # [B, H, action_dim]
        sample: bool = True,
    ) -> list[RSSMState]:
        """Roll out imagined trajectory of horizon H using learned prior dynamics only."""
        B, H, _ = action_sequence.shape
        imagined_states: list[RSSMState] = []
        cur_state = init_state

        for h in range(H):
            act_h = action_sequence[:, h]
            next_state = self.imagine_step(cur_state, act_h, sample=sample)
            imagined_states.append(next_state)
            cur_state = next_state

        return imagined_states
