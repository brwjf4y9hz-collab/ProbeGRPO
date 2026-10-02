"""本模块在 verl v1 训练器计算优势前读取每条 episode 的 probe sidecar，校验 trajectory、response mask、token 坐标后再打包注入。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from probegrpo.advantage import ProbeCredit
from probegrpo.batching import PackedProbeBatch, pack_probe_credits
from probegrpo.integration.verl import apply_to_dataproto


def pack_sidecar_probes(
    batch_keys: Sequence[str],
    response_masks: Sequence[Sequence[int]],
    *,
    sidecar_dir: Path,
    token_count: int,
    budget: int,
) -> tuple[PackedProbeBatch, dict[str, int]]:
    """将每条保存的 episode 匹配到一条补齐后的 verl 行，并校验 token 坐标。

    初始训练接入限定每个 episode 对应一个 TransferQueue 行。sidecar 缺失、轨迹
    ID 不匹配、mask 不一致或坐标落在非助手 token 上时立即报错，避免 credit
    错分给另一条 rollout。返回的统计量区分未抽中、失败和有效 probe。
    """

    if budget < 0:
        raise ValueError("probe budget must be non-negative")
    if len(batch_keys) != len(response_masks):
        raise ValueError("batch keys and response masks have different lengths")
    if budget == 0:
        return (
            pack_probe_credits([], len(batch_keys), token_count, 0),
            {
                "probe/attempted": 0,
                "probe/valid": 0,
                "probe/failed": 0,
                "probe/skipped": 0,
                "probe/unselected": len(batch_keys),
                "probe/extra_tokens": 0,
            },
        )

    # 移除 verl 为 span 添加的末尾编号，再检查每个 episode 是否只对应一行。
    trajectory_ids = [key.rsplit("_", 1)[0] for key in batch_keys]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("ProbeGRPO v1 hook does not support multi-span episodes")

    credits: list[ProbeCredit] = []
    attempted = failed = unselected = extra_tokens = 0
    for sample_index, trajectory_id in enumerate(trajectory_ids):
        filename = hashlib.sha256(trajectory_id.encode()).hexdigest() + ".json"
        path = sidecar_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing probe episode sidecar: {path}")
        episode = json.loads(path.read_text())
        if episode.get("trajectory_id") != trajectory_id:
            raise ValueError(f"sidecar trajectory ID mismatch: {path}")

        # sidecar 可短于 padding 后的序列，但有效区必须完全相等，padding 区必须全为 0。
        recorded_mask = episode["stream"]["response_mask"]
        actual_mask = list(response_masks[sample_index])
        if (
            len(recorded_mask) > token_count
            or len(actual_mask) != token_count
            or actual_mask[: len(recorded_mask)] != recorded_mask
            or any(actual_mask[len(recorded_mask) :])
        ):
            raise ValueError(f"response mask mismatch for {trajectory_id}")

        probe = episode.get("debug_probe")
        if not probe:
            unselected += 1
            continue
        attempted += 1
        cost = int(probe.get("additional_rollout_tokens", 0))
        if cost < 0:
            raise ValueError(f"negative probe token cost for {trajectory_id}")
        extra_tokens += cost
        if probe.get("skipped_reason"):
            failed += 1
            continue
        anchor_turn_id = probe["anchor_turn_id"]
        turns = episode["turns"]
        if not isinstance(anchor_turn_id, int) or not 0 <= anchor_turn_id < len(turns):
            raise ValueError(f"invalid anchor turn for {trajectory_id}")
        if (
            probe.get("anchor_id") != f"{trajectory_id}:{anchor_turn_id}"
            or not probe.get("state_match")
        ):
            raise ValueError(f"probe anchor state mismatch for {trajectory_id}")
        indices = tuple(turns[anchor_turn_id]["token_indices"])
        if not indices or any(
            not isinstance(index, int)
            or not 0 <= index < len(recorded_mask)
            or recorded_mask[index] != 1
            for index in indices
        ):
            raise ValueError(f"probe selected non-assistant tokens for {trajectory_id}")
        delta = float(probe["delta"])
        if not math.isfinite(delta):
            raise ValueError(f"non-finite probe delta for {trajectory_id}")
        credits.append(ProbeCredit(sample_index, indices, delta))

    packed = pack_probe_credits(credits, len(batch_keys), token_count, budget)
    return packed, {
        "probe/attempted": attempted,
        "probe/valid": len(credits),
        "probe/failed": failed,
        "probe/skipped": failed,
        "probe/unselected": unselected,
        "probe/extra_tokens": extra_tokens,
    }


def apply_sidecar_probe_credits(
    data: Any,
    batch_keys: Sequence[str],
    *,
    sidecar_dir: str | Path,
    budget: int = 1,
    lambda_coef: float = 0.5,
) -> tuple[Any, dict[str, int | float]]:
    """将稠密 probe 字段写入补齐后的 DataProto，再塑形其优势张量。"""

    if lambda_coef < 0:
        raise ValueError("probe lambda must be non-negative")
    if budget == 0:
        return data, {
            "probe/attempted": 0,
            "probe/valid": 0,
            "probe/failed": 0,
            "probe/skipped": 0,
            "probe/unselected": len(batch_keys),
            "probe/extra_tokens": 0,
            "probe/changed_tokens": 0,
            "probe/credit_abs_sum": 0.0,
        }

    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU runtime only
        raise RuntimeError("The verl probe hook requires PyTorch") from error

    # 先用 verl 的真实 batch 张量作为 token 长度和掩码来源，不依据文本重算坐标。
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    if advantages.ndim != 2 or response_mask.shape != advantages.shape:
        raise ValueError("expected matching [batch, token] advantages and response masks")
    packed, metrics = pack_sidecar_probes(
        batch_keys,
        response_mask.tolist(),
        sidecar_dir=Path(sidecar_dir),
        token_count=advantages.shape[1],
        budget=budget,
    )
    if lambda_coef == 0:
        metrics["probe/changed_tokens"] = 0
        metrics["probe/credit_abs_sum"] = 0.0
        return data, metrics
    device = advantages.device
    data.batch["probe_turn_masks"] = torch.tensor(
        packed.probe_turn_masks, dtype=torch.bool, device=device
    )
    data.batch["probe_deltas"] = torch.tensor(
        packed.probe_deltas, dtype=advantages.dtype, device=device
    )
    data.batch["probe_valid"] = torch.tensor(
        packed.probe_valid, dtype=torch.bool, device=device
    )
    before = advantages.clone()
    data = apply_to_dataproto(data, lambda_coef=lambda_coef)
    difference = data.batch["advantages"] - before
    metrics["probe/changed_tokens"] = int(torch.count_nonzero(difference).item())
    metrics["probe/credit_abs_sum"] = float(difference.abs().sum().item())
    return data, metrics
