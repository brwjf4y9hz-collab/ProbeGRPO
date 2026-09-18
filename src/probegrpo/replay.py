"""Deterministic environment replay primitives."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .types import ReplayState


def canonical_state_hash(state: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible environment state."""

    payload = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"State value is not JSON serializable: {type(value)!r}")


class ReplayableEnv(ABC):
    """Minimal contract required for exact counterfactual prefix replay.

    Implementations must make ``reset(task_id, seed)`` deterministic. The default ``replay``
    method intentionally re-executes public actions instead of relying on unsafe object copies.
    """

    @abstractmethod
    def reset(self, task_id: str, seed: int) -> ReplayState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> ReplayState:
        raise NotImplementedError

    def replay(self, task_id: str, seed: int, action_prefix: Sequence[str]) -> ReplayState:
        state = self.reset(task_id=task_id, seed=seed)
        for index, action in enumerate(action_prefix):
            if state.terminal:
                raise ReplayError(
                    f"Prefix contains action {index} after terminal state for task {task_id!r}"
                )
            state = self.step(action)
        return state


class ReplayError(RuntimeError):
    """Raised when an action prefix cannot be replayed safely."""


def state_from_mapping(
    observation: str,
    raw_state: Mapping[str, Any],
    legal_actions: Sequence[str] = (),
    terminal: bool = False,
    reward: float = 0.0,
) -> ReplayState:
    """Convenience helper for adapters around third-party environments."""

    return ReplayState(
        observation=observation,
        state_hash=canonical_state_hash(raw_state),
        legal_actions=tuple(legal_actions),
        terminal=terminal,
        reward=float(reward),
        metadata={"raw_state": dict(raw_state)},
    )
