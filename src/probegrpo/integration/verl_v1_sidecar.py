"""Load AgentLoop probe sidecars before verl v1 writes advantages to TransferQueue."""

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
    """Match one saved episode to each padded verl row and validate token coordinates.

    The initial training gate supports one TransferQueue row per episode. Failing on a
    missing or mismatched sidecar prevents credit from being assigned to another rollout.
    """

    if budget < 0:
        raise ValueError("probe budget must be non-negative")
    if len(batch_keys) != len(response_masks):
        raise ValueError("batch keys and response masks have different lengths")
    if budget == 0:
        return (
            pack_probe_credits([], len(batch_keys), token_count, 0),
            {"probe/valid": 0, "probe/skipped": 0, "probe/extra_tokens": 0},
        )

    trajectory_ids = [key.rsplit("_", 1)[0] for key in batch_keys]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("ProbeGRPO v1 hook does not support multi-span episodes")

    credits: list[ProbeCredit] = []
    skipped = extra_tokens = 0
    for sample_index, trajectory_id in enumerate(trajectory_ids):
        filename = hashlib.sha256(trajectory_id.encode()).hexdigest() + ".json"
        path = sidecar_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing probe episode sidecar: {path}")
        episode = json.loads(path.read_text())
        if episode.get("trajectory_id") != trajectory_id:
            raise ValueError(f"sidecar trajectory ID mismatch: {path}")

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
        if not probe or probe.get("skipped_reason"):
            skipped += 1
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
        cost = int(probe["additional_rollout_tokens"])
        if cost < 0:
            raise ValueError(f"negative probe token cost for {trajectory_id}")
        extra_tokens += cost
        credits.append(ProbeCredit(sample_index, indices, delta))

    packed = pack_probe_credits(credits, len(batch_keys), token_count, budget)
    return packed, {
        "probe/valid": len(credits),
        "probe/skipped": skipped,
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
    """Add dense probe fields to a padded DataProto, then shape its advantages."""

    if lambda_coef < 0:
        raise ValueError("probe lambda must be non-negative")
    if budget == 0 or lambda_coef == 0:
        return data, {"probe/valid": 0, "probe/skipped": 0, "probe/extra_tokens": 0}

    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU runtime only
        raise RuntimeError("The verl probe hook requires PyTorch") from error

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
