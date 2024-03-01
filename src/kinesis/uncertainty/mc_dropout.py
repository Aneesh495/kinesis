"""Monte Carlo Dropout epistemic uncertainty estimator for neural models."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MCDropoutOutput:
    """Output from Monte Carlo Dropout evaluation."""
    mean: torch.Tensor
    epistemic_variance: torch.Tensor
    epistemic_std: torch.Tensor
    samples: torch.Tensor  # [N, B, ...]


class MCDropoutEstimator:
    """Estimates epistemic uncertainty via stochastic Monte Carlo forward dropout passes."""

    def __init__(self, model: nn.Module, num_samples: int = 20) -> None:
        self.model = model
        self.num_samples = num_samples

    def _enable_dropout(self) -> None:
        """Force all Dropout modules into training mode while keeping others in eval mode."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                module.train()

    def estimate(self, *inputs: torch.Tensor) -> MCDropoutOutput:
        """Run stochastic forward passes with dropout active and compute sample statistics."""
        self.model.eval()
        self._enable_dropout()

        samples_list = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = self.model(*inputs)
                if isinstance(out, tuple):
                    out = out[0]
                samples_list.append(out.unsqueeze(0))

        # Stack into [N, B, ...]
        samples = torch.cat(samples_list, dim=0)

        mean = torch.mean(samples, dim=0)
        variance = torch.var(samples, dim=0, unbiased=True)
        std = torch.sqrt(variance + 1e-8)

        return MCDropoutOutput(
            mean=mean,
            epistemic_variance=variance,
            epistemic_std=std,
            samples=samples,
        )
