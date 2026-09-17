"""Thin torch adapter for RAGEN/verl trainer batches.

This module deliberately imports torch lazily so the core project and CPU smoke tests have no heavy
dependencies. RAGEN's trainer should call ``apply_probe_credits_tensor`` after computing standard
GRPO advantages and before the actor update.
"""

from __future__ import annotations

from typing import Any


def apply_probe_credits_tensor(
    advantages: Any,
    probe_turn_masks: Any,
    probe_deltas: Any,
    probe_valid: Any,
    lambda_coef: float = 0.5,
    epsilon: float = 1e-8,
) -> Any:
    """Blend probe deltas into `[batch, token]` GRPO advantages.

    Expected shapes:
      - advantages: `[batch, tokens]`
      - probe_turn_masks: `[batch, anchors, tokens]`
      - probe_deltas: `[batch, anchors]`
      - probe_valid: `[batch, anchors]`
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only in GPU environments
        raise RuntimeError("The RAGEN adapter requires the 'ragen' extra and PyTorch") from error

    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")
    if advantages.ndim != 2 or probe_turn_masks.ndim != 3:
        raise ValueError("advantages must be 2D and probe_turn_masks must be 3D")
    if probe_deltas.ndim != 2 or probe_valid.ndim != 2:
        raise ValueError("probe_deltas and probe_valid must be 2D")
    if probe_turn_masks.shape[:2] != probe_deltas.shape:
        raise ValueError("mask anchor dimensions must match probe_deltas")
    if probe_deltas.shape != probe_valid.shape:
        raise ValueError("probe_deltas and probe_valid must have identical shape")
    if probe_turn_masks.shape[0] != advantages.shape[0]:
        raise ValueError("batch dimensions do not match")
    if probe_turn_masks.shape[2] != advantages.shape[1]:
        raise ValueError("token dimensions do not match")
    if lambda_coef == 0:
        return advantages

    valid = probe_valid.bool()
    selected = probe_deltas[valid].to(dtype=advantages.dtype)
    if selected.numel() == 0:
        return advantages
    if selected.numel() == 1:
        normalized_selected = torch.sign(selected)
    else:
        std = selected.std(unbiased=False)
        if std <= epsilon:
            normalized_selected = torch.zeros_like(selected)
        else:
            normalized_selected = (selected - selected.mean()) / std

    normalized = torch.zeros_like(probe_deltas, dtype=advantages.dtype)
    normalized[valid] = normalized_selected
    adjustment = (
        normalized.unsqueeze(-1) * probe_turn_masks.to(dtype=advantages.dtype)
    ).sum(dim=1)
    return advantages + float(lambda_coef) * adjustment


def apply_to_dataproto(data: Any, lambda_coef: float = 0.5) -> Any:
    """Mutate a verl ``DataProto`` batch in the same style as RAGEN's trainer utilities."""

    required = ("advantages", "probe_turn_masks", "probe_deltas", "probe_valid")
    missing = [name for name in required if name not in data.batch]
    if missing:
        raise KeyError(f"ProbeGRPO batch is missing fields: {', '.join(missing)}")
    data.batch["advantages"] = apply_probe_credits_tensor(
        advantages=data.batch["advantages"],
        probe_turn_masks=data.batch["probe_turn_masks"],
        probe_deltas=data.batch["probe_deltas"],
        probe_valid=data.batch["probe_valid"],
        lambda_coef=lambda_coef,
    )
    return data

