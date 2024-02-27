"""Vector Quantized Latent State Dynamics (VQ-RSSM) with straight-through gradient estimation."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    """Output from vector quantizer."""
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    """Vector Quantizer with straight-through estimator and commitment loss."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> VQOutput:
        """Quantize continuous inputs [B, D] or [B, T, D] to nearest codebook vector."""
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z * e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(input_shape)

        # Perplexity calculation for codebook utilization tracking
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through gradient: backprop past quantization directly to input
        quantized_st = inputs + (quantized - inputs).detach()

        return VQOutput(
            quantized=quantized_st,
            indices=encoding_indices.view(input_shape[:-1]),
            loss=loss,
            perplexity=perplexity,
        )


class VectorQuantizedRSSM(nn.Module):
    """Recurrent State-Space Model with discrete Vector Quantized stochastic states."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 32,
        deter_dim: int = 64,
        num_embeddings: int = 128,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Vector quantizer for stochastic representation
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Deterministic recurrent GRU cell: input is [z_quantized, action]
        self.gru = nn.GRUCell(embedding_dim + action_dim, deter_dim)

        # Prior projection: h_t -> continuous z_prior -> quantizer
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Posterior projection: [h_t, embed_t] -> continuous z_post -> quantizer
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Reconstruction heads
        feature_dim = deter_dim + embedding_dim
        self.obs_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce zero initial deterministic state and discrete latent."""
        h0 = torch.zeros(batch_size, self.deter_dim, device=device)
        z0 = torch.zeros(batch_size, self.embedding_dim, device=device)
        return h0, z0

    def observe_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, VQOutput, VQOutput]:
        """Single observe step updating GRU and computing posterior vs prior codebook outputs."""
        gru_in = torch.cat([z_prev, action], dim=-1)
        h_curr = self.gru(gru_in, h_prev)

        # Prior
        cont_prior = self.prior_net(h_curr)
        prior_vq = self.quantizer(cont_prior)

        # Posterior
        embed = self.encoder(observation)
        post_in = torch.cat([h_curr, embed], dim=-1)
        cont_post = self.post_net(post_in)
        post_vq = self.quantizer(cont_post)

        return h_curr, post_vq, prior_vq

    def imagine_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, VQOutput]:
        """Single imagination step using prior network and codebook quantization."""
        gru_in = torch.cat([z_prev, action], dim=-1)
        h_curr = self.gru(gru_in, h_prev)
        cont_prior = self.prior_net(h_curr)
        prior_vq = self.quantizer(cont_prior)
        return h_curr, prior_vq
