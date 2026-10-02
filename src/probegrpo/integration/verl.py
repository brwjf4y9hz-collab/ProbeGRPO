"""本模块把框架无关的稀疏 probe credit 转换为 PyTorch 张量运算，并适配 verl DataProto 的优势字段。张量轴顺序和 mask 决定实际受影响的 token。"""

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
    """将 probe 奖励差混入 ``[batch, token]`` 的 GRPO 优势。

    预期形状：
      - advantages: ``[batch, tokens]``
      - probe_turn_masks: ``[batch, anchors, tokens]``
      - probe_deltas: ``[batch, anchors]``
      - probe_valid: ``[batch, anchors]``
    有效 delta 采用与无框架实现一致的零保持缩放，再沿 anchor 维求和；mask
    决定每个 delta 写入哪些动作 token。输入形状或有限性不满足约定时立即报错。
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only in GPU environments
        raise RuntimeError("The verl adapter requires a GPU runtime with PyTorch") from error

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

    # 只对有效 probe 计算缩放上界；跳过项的占位 delta 不参与归一化。
    valid = probe_valid.bool()
    selected = probe_deltas[valid].to(dtype=advantages.dtype)
    if selected.numel() == 0:
        return advantages
    if not torch.isfinite(selected).all():
        raise ValueError("probe deltas must be finite")
    largest = selected.abs().max()
    if largest <= epsilon:
        normalized_selected = torch.zeros_like(selected)
    else:
        scale = largest.clamp(min=1.0)
        normalized_selected = selected / scale

    normalized = torch.zeros_like(probe_deltas, dtype=advantages.dtype)
    normalized[valid] = normalized_selected
    # 一个 token 若属于多个有效 anchor，则累加各 anchor 的局部调整量。
    adjustment = (
        normalized.unsqueeze(-1) * probe_turn_masks.to(dtype=advantages.dtype)
    ).sum(dim=1)
    return advantages + float(lambda_coef) * adjustment


def apply_to_dataproto(data: Any, lambda_coef: float = 0.5) -> Any:
    """把 probe credit 加入 verl ``DataProto`` 的优势字段并返回 batch。"""

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
