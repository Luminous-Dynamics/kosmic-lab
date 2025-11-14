"""
Correct FGSM (Fast Gradient Sign Method) implementation.

FGSM: x' = x + ε * sign(∇_x L(x,y))

Gradients are taken w.r.t. observation tensor, NOT K-Index.
K-Index is never backpropagated through.
"""

import torch
import torch.nn as nn
from typing import Tuple


def fgsm_observation(
    model: nn.Module, obs: torch.Tensor, target: torch.Tensor, loss_fn: nn.Module, eps: float
) -> torch.Tensor:
    """
    Apply FGSM adversarial perturbation to observations.

    Args:
        model: Policy network
        obs: Observation tensor (requires_grad will be set)
        target: Target labels/actions
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        eps: Perturbation magnitude

    Returns:
        Adversarially perturbed observations

    Example:
        >>> model = PolicyNet()
        >>> obs = torch.randn(32, 4)
        >>> target = torch.randint(0, 2, (32,))
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> adv_obs = fgsm_observation(model, obs, target, loss_fn, eps=0.1)
    """
    # Clone and enable gradients
    obs = obs.clone().detach().requires_grad_(True)

    # Forward pass
    logits = model(obs)
    loss = loss_fn(logits, target)

    # Backward pass to get gradients
    loss.backward()
    grad = obs.grad

    # Apply FGSM perturbation
    adv_obs = obs + eps * torch.sign(grad)

    return adv_obs.detach()


@torch.no_grad()
def sanity_check_loss_increases(
    model: nn.Module, obs: torch.Tensor, target: torch.Tensor, loss_fn: nn.Module, eps: float
) -> Tuple[float, float]:
    """
    Verify FGSM perturbation increases task loss.

    Args:
        model: Policy network
        obs: Clean observations
        target: Target labels/actions
        loss_fn: Loss function
        eps: Perturbation magnitude

    Returns:
        (base_loss, adversarial_loss) tuple

    Raises:
        AssertionError: If adversarial loss doesn't increase

    Example:
        >>> base, adv = sanity_check_loss_increases(model, obs, target, loss_fn, 0.1)
        >>> assert adv >= base, "FGSM should increase task loss"
    """
    model.eval()

    # Base loss (clean)
    logits_clean = model(obs)
    base_loss = loss_fn(logits_clean, target).item()

    # Adversarial loss (perturbed)
    # Need to re-enable gradients for perturbation
    obs_for_attack = obs.clone().detach().requires_grad_(True)
    logits = model(obs_for_attack)
    loss = loss_fn(logits, target)
    loss.backward()

    adv_obs = obs_for_attack + eps * torch.sign(obs_for_attack.grad)
    logits_adv = model(adv_obs.detach())
    adv_loss = loss_fn(logits_adv, target).item()

    return base_loss, adv_loss


def fgsm_batch(
    model: nn.Module,
    obs_batch: torch.Tensor,
    target_batch: torch.Tensor,
    loss_fn: nn.Module,
    eps: float,
    verify: bool = True,
) -> torch.Tensor:
    """
    Apply FGSM to a batch with optional sanity check.

    Args:
        model: Policy network
        obs_batch: Batch of observations [batch_size, obs_dim]
        target_batch: Batch of targets [batch_size]
        loss_fn: Loss function
        eps: Perturbation magnitude
        verify: If True, verify loss increases (default: True)

    Returns:
        Adversarially perturbed observation batch

    Raises:
        AssertionError: If verify=True and loss doesn't increase
    """
    if verify:
        base, adv = sanity_check_loss_increases(model, obs_batch, target_batch, loss_fn, eps)
        assert adv >= base, (
            f"FGSM sanity check failed: " f"adv loss {adv:.4f} < base loss {base:.4f}"
        )

    return fgsm_observation(model, obs_batch, target_batch, loss_fn, eps)
