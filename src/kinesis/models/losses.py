"""Balanced KL divergence, free bits, and multi-task latent loss objectives."""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F

from kinesis.models.rssm import RSSMState


@dataclass(frozen=True)
class WorldModelLossResult:
    """Detailed breakdown of world model training loss components and diagnostics."""
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    reward_loss: torch.Tensor
    discount_loss: torch.Tensor
    constraint_loss: torch.Tensor
    prior_entropy: torch.Tensor
    post_entropy: torch.Tensor
    latent_usage_ratio: float


def compute_gaussian_kl(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """Analytical Kullback-Leibler divergence between two diagonal Gaussians:

    KL(q || p) = sum_i [ log(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2) - 0.5 ]
    """
    var_post = post_std.pow(2)
    var_prior = prior_std.pow(2)

    t1 = torch.log(prior_std / post_std)
    t2 = (var_post + (post_mean - prior_mean).pow(2)) / (2.0 * var_prior)
    kl_per_dim = t1 + t2 - 0.5
    return kl_per_dim.sum(dim=-1)


def compute_balanced_kl_loss(
    prior_states: list[RSSMState],
    post_states: list[RSSMState],
    alpha: float = 0.8,
    free_bits: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute balanced KL loss with free bits to prevent posterior collapse:

    L_KL = alpha * KL(sg(q) || p) + (1 - alpha) * KL(q || sg(p))
    """
    kl_prior_parts = []
    kl_post_parts = []
    prior_entropies = []
    post_entropies = []
    active_dims = []

    for prior_s, post_s in zip(prior_states, post_states):
        # Stop-gradient versions
        q_mean_sg, q_std_sg = post_s.mean.detach(), post_s.std.detach()
        p_mean_sg, p_std_sg = prior_s.mean.detach(), prior_s.std.detach()

        # Prior tracking loss (alpha weight)
        kl_prior = compute_gaussian_kl(q_mean_sg, q_std_sg, prior_s.mean, prior_s.std)
        # Posterior regularization loss (1 - alpha weight)
        kl_post = compute_gaussian_kl(post_s.mean, post_s.std, p_mean_sg, p_std_sg)

        # Apply free-bits floor
        kl_prior = torch.clamp(kl_prior, min=free_bits)
        kl_post = torch.clamp(kl_post, min=free_bits)

        kl_prior_parts.append(kl_prior.mean())
        kl_post_parts.append(kl_post.mean())

        # Entropies: 0.5 * log(2 * pi * e * sigma^2)
        p_ent = 0.5 + 0.5 * torch.log(2.0 * torch.pi * prior_s.std.pow(2))
        q_ent = 0.5 + 0.5 * torch.log(2.0 * torch.pi * post_s.std.pow(2))
        prior_entropies.append(p_ent.sum(dim=-1).mean())
        post_entropies.append(q_ent.sum(dim=-1).mean())

        # Active dimension check: variance across batch > 0.05
        batch_var = torch.var(post_s.mean, dim=0)
        active_ratio = float((batch_var > 0.05).float().mean().item())
        active_dims.append(active_ratio)

    loss_prior = torch.stack(kl_prior_parts).mean()
    loss_post = torch.stack(kl_post_parts).mean()
    total_kl_loss = alpha * loss_prior + (1.0 - alpha) * loss_post

    mean_p_ent = torch.stack(prior_entropies).mean()
    mean_q_ent = torch.stack(post_entropies).mean()
    mean_usage = float(sum(active_dims) / len(active_dims))

    return total_kl_loss, mean_p_ent, mean_q_ent, mean_usage
