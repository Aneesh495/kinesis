"""Observation encoders and reconstruction decoders for proprioceptive and visual states."""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinesis.contracts import ObservationSpec


class ProprioceptiveEncoder(nn.Module):
    """Multi-layer perceptron encoding permitted public observation vectors into latent embeddings."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [..., input_dim] -> [..., embed_dim]."""
        return self.net(x)


class ProprioceptiveDecoder(nn.Module):
    """Decodes latent features [h_t, z_t] back to observation space for reconstruction learning."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_d = feature_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode [..., feature_dim] -> [..., output_dim]."""
        return self.net(features)
