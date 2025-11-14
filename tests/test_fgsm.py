"""
Unit tests for FGSM implementation.

These tests verify:
1. FGSM increases task loss (sanity check)
2. Perturbation magnitude is bounded by epsilon
3. Gradient computation is correct
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from fre.attacks.fgsm import fgsm_observation, sanity_check_loss_increases, fgsm_batch


class SimplePolicy(nn.Module):
    """Simple policy network for testing."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, act_dim))

    def forward(self, x):
        return self.fc(x)


def test_fgsm_increases_loss():
    """Test that FGSM perturbation increases task loss."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.1

    # Compute losses
    base_loss, adv_loss = sanity_check_loss_increases(model, obs, target, loss_fn, eps)

    # Verify adversarial loss >= base loss
    assert (
        adv_loss >= base_loss - 1e-6
    ), f"FGSM should increase task loss, but adv={adv_loss:.4f} < base={base_loss:.4f}"


def test_fgsm_perturbation_magnitude():
    """Test that perturbation magnitude is bounded by epsilon."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.1

    # Apply FGSM
    adv_obs = fgsm_observation(model, obs, target, loss_fn, eps)

    # Check perturbation magnitude
    delta = (adv_obs - obs).abs()
    max_delta = delta.max().item()

    assert max_delta <= eps + 1e-6, f"Perturbation magnitude {max_delta:.6f} exceeds epsilon {eps}"


def test_fgsm_gradient_direction():
    """Test that FGSM perturbs in the sign of the gradient."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4, requires_grad=True)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.1

    # Compute gradient manually
    logits = model(obs)
    loss = loss_fn(logits, target)
    loss.backward()
    grad_sign = obs.grad.sign()

    # Apply FGSM
    obs_detached = obs.clone().detach()
    adv_obs = fgsm_observation(model, obs_detached, target, loss_fn, eps)

    # Check perturbation aligns with gradient sign
    delta = adv_obs - obs_detached
    expected_delta = eps * grad_sign

    # Allow small numerical errors
    alignment = ((delta.sign() == expected_delta.sign()).float().mean()).item()

    assert (
        alignment >= 0.99
    ), f"FGSM perturbation should align with gradient sign, but only {alignment*100:.1f}% match"


def test_fgsm_batch_with_verify():
    """Test fgsm_batch with sanity check enabled."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.1

    # Should not raise (adversarial loss should be >= base loss)
    adv_obs = fgsm_batch(model, obs, target, loss_fn, eps, verify=True)

    assert adv_obs.shape == obs.shape, "Output shape should match input"


def test_fgsm_zero_epsilon():
    """Test that epsilon=0 produces no perturbation."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.0

    # Apply FGSM with eps=0
    adv_obs = fgsm_observation(model, obs, target, loss_fn, eps)

    # Check no perturbation
    assert torch.allclose(adv_obs, obs, atol=1e-6), "Epsilon=0 should produce no perturbation"


def test_fgsm_deterministic():
    """Test that FGSM is deterministic (same input → same output)."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    model.eval()  # Disable dropout/batchnorm randomness

    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    eps = 0.1

    # Apply FGSM twice
    adv_obs_1 = fgsm_observation(model, obs, target, loss_fn, eps)
    adv_obs_2 = fgsm_observation(model, obs, target, loss_fn, eps)

    # Check determinism
    assert torch.allclose(adv_obs_1, adv_obs_2, atol=1e-6), "FGSM should be deterministic"


def test_fgsm_different_epsilon():
    """Test that larger epsilon produces larger perturbations."""
    # Setup
    model = SimplePolicy(obs_dim=4, act_dim=2)
    obs = torch.randn(32, 4)
    target = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()

    eps_small = 0.01
    eps_large = 0.1

    # Apply FGSM with different epsilons
    adv_obs_small = fgsm_observation(model, obs, target, loss_fn, eps_small)
    adv_obs_large = fgsm_observation(model, obs, target, loss_fn, eps_large)

    # Compute perturbation norms
    delta_small = (adv_obs_small - obs).norm().item()
    delta_large = (adv_obs_large - obs).norm().item()

    assert delta_large > delta_small, (
        f"Larger epsilon should produce larger perturbations, but "
        f"eps={eps_large} → Δ={delta_large:.4f}, eps={eps_small} → Δ={delta_small:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
