"""Framework-independent rollout traces and ``TurnRecord`` extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import TurnRecord


@dataclass(frozen=True)
class RawAssistantTurn:
    """Minimal data captured for one assistant action during a rollout.

    ``token_indices`` refers to positions in the padded trainer sequence. Entropies and top-two
    log-probabilities are aligned one-to-one with those positions.
    """

    action: str
    state_hash: str
    token_indices: Tuple[int, ...]
    token_entropies: Tuple[float, ...]
    top1_logprobs: Tuple[float, ...]
    top2_logprobs: Tuple[float, ...]
    legal_actions: Tuple[str, ...]
    action_valid: bool = True
    terminal: bool = False


@dataclass(frozen=True)
class TrajectoryTrace:
    """One completed trajectory before it is converted into scheduler records."""

    trajectory_id: str
    task_id: str
    seed: int
    token_count: int
    max_horizon: int
    final_reward: float
    turns: Tuple[RawAssistantTurn, ...]


def extract_turn_records(trace: TrajectoryTrace) -> List[TurnRecord]:
    """Validate a trajectory and derive one replayable record per assistant turn."""

    _validate_trace_header(trace)
    suffix_token_counts = _suffix_token_counts(trace.turns)
    action_prefix: List[str] = []
    invalid_actions_before = 0
    used_token_indices = set()
    records: List[TurnRecord] = []

    for turn_id, raw_turn in enumerate(trace.turns):
        _validate_raw_turn(raw_turn, turn_id, trace.token_count, used_token_indices)
        records.append(
            TurnRecord(
                trajectory_id=trace.trajectory_id,
                task_id=trace.task_id,
                seed=int(trace.seed),
                turn_id=turn_id,
                action=str(raw_turn.action),
                action_prefix=tuple(action_prefix),
                state_hash=str(raw_turn.state_hash),
                token_indices=tuple(raw_turn.token_indices),
                mean_entropy=_mean(raw_turn.token_entropies),
                logprob_margin=_mean(
                    tuple(
                        float(top1) - float(top2)
                        for top1, top2 in zip(
                            raw_turn.top1_logprobs,
                            raw_turn.top2_logprobs,
                        )
                    )
                ),
                invalid_actions_before=invalid_actions_before,
                legal_actions=tuple(str(action) for action in raw_turn.legal_actions),
                max_horizon=int(trace.max_horizon),
                suffix_tokens=suffix_token_counts[turn_id],
                final_reward=float(trace.final_reward),
                terminal=bool(raw_turn.terminal),
                metadata={"action_valid": bool(raw_turn.action_valid)},
            )
        )
        used_token_indices.update(raw_turn.token_indices)
        action_prefix.append(str(raw_turn.action))
        if not raw_turn.action_valid:
            invalid_actions_before += 1

    return records


def _validate_trace_header(trace: TrajectoryTrace) -> None:
    if not trace.trajectory_id:
        raise ValueError("trajectory_id must not be empty")
    if not trace.task_id:
        raise ValueError("task_id must not be empty")
    if trace.token_count <= 0:
        raise ValueError("token_count must be positive")
    if trace.max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    if len(trace.turns) > trace.max_horizon:
        raise ValueError("number of turns exceeds max_horizon")


def _validate_raw_turn(
    turn: RawAssistantTurn,
    turn_id: int,
    token_count: int,
    used_token_indices: set,
) -> None:
    if not turn.action:
        raise ValueError(f"turn {turn_id} action must not be empty")
    if not turn.state_hash:
        raise ValueError(f"turn {turn_id} state_hash must not be empty")
    if not turn.token_indices:
        raise ValueError(f"turn {turn_id} must contain at least one assistant token")

    aligned_lengths = {
        len(turn.token_indices),
        len(turn.token_entropies),
        len(turn.top1_logprobs),
        len(turn.top2_logprobs),
    }
    if len(aligned_lengths) != 1:
        raise ValueError(f"turn {turn_id} token statistics must have identical lengths")
    if len(set(turn.token_indices)) != len(turn.token_indices):
        raise ValueError(f"turn {turn_id} contains duplicate token indices")

    for token_index in turn.token_indices:
        if not 0 <= token_index < token_count:
            raise ValueError(
                f"turn {turn_id} token index {token_index} is outside [0, {token_count})"
            )
        if token_index in used_token_indices:
            raise ValueError(f"token index {token_index} belongs to more than one turn")

    for top1, top2 in zip(turn.top1_logprobs, turn.top2_logprobs):
        if float(top1) < float(top2):
            raise ValueError(f"turn {turn_id} top1 log-probability is below top2")


def _suffix_token_counts(turns: Tuple[RawAssistantTurn, ...]) -> List[int]:
    counts = [0] * len(turns)
    running_total = 0
    for index in range(len(turns) - 1, -1, -1):
        running_total += len(turns[index].token_indices)
        counts[index] = running_total
    return counts


def _mean(values: Tuple[float, ...]) -> float:
    return sum(float(value) for value in values) / len(values)
