"""Deep Evidential Regression for closed-form epistemic and aleatoric uncertainty decomposition."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvidentialOutput:
    """Predicted parameters of Normal-Inverse-Gamma distribution."""
    gamma: torch.Tensor  # Mean prediction [B, D]
    v: torch.Tensor      # Evidence precision [B, D] (v > 0)
    alpha: torch.Tensor  # Shape parameter [B, D] (alpha > 1)
    beta: torch.Tensor   # Scale parameter [B, D] (beta > 0)
    aleatoric_variance: torch.Tensor  # beta / (alpha - 1)
    epistemic_variance: torch.Tensor  # beta / (v * (alpha - 1))


class EvidentialRegressionHead(nn.Module):
    """Outputs conjugate Normal-Inverse-Gamma distribution parameters for continuous targets."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Predicts [gamma, v, alpha, beta] for each target dimension
        self.dense = nn.Linear(in_features, 4 * out_features)

    def forward(self, x: torch.Tensor) -> EvidentialOutput:
        """Map feature embeddings to NIG parameters with positivity constraints."""
        out = self.dense(x)
        gamma, raw_v, raw_alpha, raw_beta = torch.chunk(out, 4, dim=-1)

        # Enforce support constraints: v > 0, alpha > 1, beta > 0
        v = F.softplus(raw_v) + 1e-6
        alpha = F.softplus(raw_alpha) + 1.0 + 1e-6
        beta = F.softplus(raw_beta) + 1e-6

        # Variance decomposition
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (v * (alpha - 1.0))

        return EvidentialOutput(
            gamma=gamma,
            v=v,
            alpha=alpha,
            beta=beta,
            aleatoric_variance=aleatoric,
            epistemic_variance=epistemic,
        )


def evidential_regression_loss(
    pred: EvidentialOutput,
    target: torch.Tensor,
    coeff: float = 0.05,
) -> torch.Tensor:
    """Negative log-likelihood of Student-t marginal plus evidence-regularizing penalty."""
    gamma, v, alpha, beta = pred.gamma, pred.v, pred.alpha, pred.beta

    # 2 * beta * (1 + v)
    omega = 2.0 * beta * (1.0 + v)

    # Student-t NLL
    diff_sq = (target - gamma) ** 2
    nll = (
        0.5 * torch.log(torch.tensor(torch.pi) / v)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(diff_sq * v + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Regularization penalty penalizing high evidence on high prediction errors
    error = torch.abs(target - gamma)
    reg = error * (2.0 * v + alpha)

    total_loss = torch.mean(nll + coeff * reg)
    return total_loss
