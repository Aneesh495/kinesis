"""Tests for world models, RSSM dynamics, encoders, decoders, and balanced latent losses."""

import pytest
import torch
import numpy as np

from kinesis.models import (
    ProprioceptiveDecoder,
    ProprioceptiveEncoder,
    RecurrentStateSpaceModel,
    RewardHead,
    WorldModel,
    compute_gaussian_kl,
)


def test_encoder_and_decoder_shapes() -> None:
    enc = ProprioceptiveEncoder(input_dim=10, embed_dim=64, hidden_dim=64)
    x = torch.randn(4, 10)
    embed = enc(x)
    assert embed.shape == (4, 64)

    dec = ProprioceptiveDecoder(feature_dim=64 + 16, output_dim=10, hidden_dim=64)
    feat = torch.randn(4, 80)
    recon = dec(feat)
    assert recon.shape == (4, 10)


def test_rssm_observe_and_imagine_steps() -> None:
    rssm = RecurrentStateSpaceModel(action_dim=2, embed_dim=64, deter_dim=128, stoch_dim=16)
    B = 3
    init_s = rssm.initial_state(B, device=torch.device("cpu"))
    act = torch.randn(B, 2)
    emb = torch.randn(B, 64)

    prior_s, post_s = rssm.observe_step(init_s, act, emb)
    assert prior_s.deter.shape == (B, 128)
    assert prior_s.stoch.shape == (B, 16)
    assert post_s.deter.shape == (B, 128)
    assert post_s.stoch.shape == (B, 16)
    assert post_s.features.shape == (B, 128 + 16)

    # Imagine step uses prior only (no observation embed required)
    imagined_s = rssm.imagine_step(post_s, act)
    assert imagined_s.features.shape == (B, 144)


def test_causal_prefix_invariance_under_future_observation_alteration() -> None:
    """Rigorous gate test: altering future observations must leave past causal belief states strictly unchanged."""
    rssm = RecurrentStateSpaceModel(action_dim=1, embed_dim=32, deter_dim=64, stoch_dim=16)
    B = 2
    T = 6
    K = 3  # Prefix split point

    embeds_original = torch.randn(B, T, 32)
    actions = torch.randn(B, T, 1)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    # Pass 1: original full sequence
    torch.manual_seed(42)
    _, post_states_1 = rssm.observe_sequence(embeds_original, actions, is_first)

    # Alter future observations at t >= K
    embeds_altered = embeds_original.clone()
    embeds_altered[:, K:] = torch.randn(B, T - K, 32) * 5.0 + 10.0

    # Pass 2: sequence with altered future
    torch.manual_seed(42)
    _, post_states_2 = rssm.observe_sequence(embeds_altered, actions, is_first)

    # Causal prefix check: belief states at t < K must match exactly
    for t in range(K):
        assert torch.allclose(post_states_1[t].deter, post_states_2[t].deter, atol=1e-6)
        assert torch.allclose(post_states_1[t].mean, post_states_2[t].mean, atol=1e-6)


def test_action_sensitivity_in_imagined_trajectories() -> None:
    """Verify that distinct action inputs drive distinct imagined latent paths (Gate RA05 prerequisite)."""
    wm = WorldModel(obs_dim=6, action_dim=2, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)
    obs = torch.randn(6)
    sensitivity = wm.check_action_sensitivity(obs)
    assert sensitivity > 0.05, f"World model is insensitive to actions! Divergence: {sensitivity}"


def test_world_model_loss_and_gradient_flow() -> None:
    wm = WorldModel(obs_dim=5, action_dim=2, embed_dim=32, deter_dim=64, stoch_dim=16, hidden_dim=64)
    B, T = 4, 8

    obs = torch.randn(B, T, 5)
    acts = torch.randn(B, T, 2)
    rews = torch.randn(B, T)
    discs = torch.ones(B, T)
    discs[:, -1] = 0.0
    costs = torch.zeros(B, T)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    loss_res = wm.compute_losses(
        observations=obs,
        actions=acts,
        rewards=rews,
        discounts=discs,
        constraint_costs=costs,
        is_first=is_first,
    )

    assert torch.isfinite(loss_res.total_loss)
    assert loss_res.recon_loss > 0.0
    assert loss_res.kl_loss > 0.0

    # Backward pass
    loss_res.total_loss.backward()

    # Check non-zero gradients on all components
    for name, param in wm.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.any(param.grad != 0), f"All-zero gradient for {name}"
