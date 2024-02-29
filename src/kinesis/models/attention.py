"""Causal Transformer world model backbone with multi-head self-attention."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerWorldModelOutput:
    """Output from Transformer world model."""
    pred_observations: torch.Tensor  # [B, T, obs_dim]
    pred_rewards: torch.Tensor       # [B, T, 1]
    latent_features: torch.Tensor    # [B, T, d_model]
    attention_weights: list[torch.Tensor]


class CausalSelfAttentionBlock(nn.Module):
    """Transformer block with causal multi-head self-attention and Pre-LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with causal masking."""
        T = x.shape[1]
        device = x.device

        # Causal mask: upper triangular with diagonal=1 is masked out
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

        norm_x = self.ln1(x)
        attn_out, attn_weights = self.mha(norm_x, norm_x, norm_x, attn_mask=causal_mask)
        x = x + attn_out

        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class CausalTransformerWorldModel(nn.Module):
    """Sequence-level causal transformer world model."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 256,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token projections
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)
        self.token_fusion = nn.Linear(2 * d_model, d_model)

        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Stacked causal attention blocks
        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Prediction heads
        self.obs_head = nn.Linear(d_model, obs_dim)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(
        self,
        observations: torch.Tensor,  # [B, T, obs_dim]
        actions: torch.Tensor,       # [B, T, action_dim]
    ) -> TransformerWorldModelOutput:
        """Process full sequence of observation and action tokens."""
        B, T, _ = observations.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        device = observations.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # Project tokens
        o_tokens = self.obs_proj(observations)
        a_tokens = self.act_proj(actions)
        x = self.token_fusion(torch.cat([o_tokens, a_tokens], dim=-1))
        x = x + self.pos_emb(positions)

        attn_weights_list = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)

        x = self.final_ln(x)

        pred_obs = self.obs_head(x)
        pred_rew = self.reward_head(x)

        return TransformerWorldModelOutput(
            pred_observations=pred_obs,
            pred_rewards=pred_rew,
            latent_features=x,
            attention_weights=attn_weights_list,
        )
