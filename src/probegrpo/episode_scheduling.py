"""Select one probe anchor inside a completed agent episode."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .agent_episode import Episode
from .schedulers import LinearUCBScheduler, RandomScheduler
from .types import Anchor, ProbeResult, TurnRecord


def episode_turn_records(episode: Episode) -> tuple[TurnRecord, ...]:
    """Build scheduler records using chosen-token surprisal available during rollout."""

    records = []
    invalid_before = 0
    response_tokens = len(episode.stream.response_mask)
    for turn_id, turn in enumerate(episode.turns):
        logprobs = turn.generated_logprobs
        surprisal = (
            -sum(float(value) for value in logprobs) / len(logprobs)
            if logprobs
            else 0.0
        )
        records.append(
            TurnRecord(
                trajectory_id=episode.trajectory_id,
                task_id=episode.task_id,
                seed=episode.seed,
                turn_id=turn_id,
                action=turn.action,
                action_prefix=turn.action_prefix,
                state_hash=turn.state_hash,
                token_indices=turn.token_indices,
                mean_entropy=surprisal,
                logprob_margin=0.0,
                invalid_actions_before=invalid_before,
                legal_actions=turn.legal_actions,
                max_horizon=episode.max_horizon,
                suffix_tokens=response_tokens - turn.token_indices[0],
                final_reward=episode.final_reward,
                terminal=False,
                metadata={
                    "uncertainty_statistic": "chosen_token_surprisal",
                    "logprobs_available": bool(logprobs),
                },
            )
        )
        if not turn.action_valid:
            invalid_before += 1
    return tuple(records)


def select_episode_anchor(
    episode: Episode,
    scheduler_name: str,
    *,
    training_update: int,
    settings: Mapping[str, Any],
    sidecar_root: Path,
) -> Anchor | None:
    candidates = episode_turn_records(episode)
    seed = _stable_seed(episode.trajectory_id)
    if scheduler_name in {"random", "random_debug"}:
        selected = RandomScheduler(seed=seed).select(candidates, 1, training_update)
    elif scheduler_name == "surprisal":
        eligible = [
            turn
            for turn in candidates
            if turn.action in turn.legal_actions
            and len(set(turn.legal_actions)) >= 2
            and turn.metadata["logprobs_available"]
        ]
        selected = (
            [
                Anchor(
                    turn=max(eligible, key=lambda turn: (turn.mean_entropy, turn.anchor_id)),
                    scheduler="surprisal",
                    score=max(turn.mean_entropy for turn in eligible),
                )
            ]
            if eligible
            else []
        )
    elif scheduler_name == "linear_ucb":
        scheduler = _linear_ucb(settings, seed)
        _observe_history(scheduler, sidecar_root, before_step=training_update)
        selected = scheduler.select(candidates, 1, training_update)
    else:
        raise ValueError(f"unsupported episode scheduler: {scheduler_name}")
    return selected[0] if selected else None


def serialized_scheduler_turn(anchor: Anchor) -> dict[str, Any]:
    payload = asdict(anchor.turn)
    payload["scheduler_label"] = anchor.scheduler
    payload["scheduler_score"] = anchor.score
    return payload


def _linear_ucb(settings: Mapping[str, Any], seed: int) -> LinearUCBScheduler:
    return LinearUCBScheduler(
        alpha=float(settings.get("ucb_alpha", 1.0)),
        warmup_updates=int(settings.get("warmup_updates", 20)),
        warmup_probes=int(settings.get("warmup_probes", 200)),
        exploration_rate=float(settings.get("exploration_rate", 0.1)),
        l2=float(settings.get("ucb_l2", 1.0)),
        seed=seed,
    )


def _observe_history(
    scheduler: LinearUCBScheduler,
    sidecar_root: Path,
    *,
    before_step: int,
) -> None:
    if not sidecar_root.is_dir():
        return
    paths = []
    for step_dir in sidecar_root.glob("step-*"):
        try:
            step = int(step_dir.name.removeprefix("step-"))
        except ValueError:
            continue
        if step < before_step:
            paths.extend(sorted(step_dir.glob("*.json")))
    for path in sorted(paths, key=str):
        payload = json.loads(path.read_text())
        probe = payload.get("debug_probe") or {}
        turn_payload = probe.get("scheduler_turn")
        if not turn_payload or not str(probe.get("scheduler", "")).startswith("linear_ucb"):
            continue
        result = _probe_result(probe)
        if result.valid and result.additional_rollout_tokens > 0:
            scheduler.observe(_turn_record(turn_payload), result)


def _turn_record(payload: Mapping[str, Any]) -> TurnRecord:
    values = dict(payload)
    values.pop("scheduler_label", None)
    values.pop("scheduler_score", None)
    for key in ("action_prefix", "token_indices", "legal_actions"):
        values[key] = tuple(values[key])
    return TurnRecord(**values)


def _probe_result(payload: Mapping[str, Any]) -> ProbeResult:
    fields = ProbeResult.__dataclass_fields__
    values = {key: payload[key] for key in fields if key in payload}
    result = ProbeResult(**values)
    if not math.isfinite(result.delta):
        raise ValueError("non-finite historical probe delta")
    return result


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")
