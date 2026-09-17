"""Stable data contracts shared by rollouts, schedulers, and trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReplayState:
    """Environment state reached after resetting or replaying an action prefix."""

    observation: str
    state_hash: str
    legal_actions: Tuple[str, ...] = ()
    terminal: bool = False
    reward: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutOutcome:
    """Result of continuing an environment from a replayed anchor."""

    reward: float
    token_count: int
    steps: int
    terminal: bool
    actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnRecord:
    """One assistant turn and the information required to replay and score it.

    ``action_prefix`` contains environment actions strictly before this turn. ``token_indices``
    identifies the assistant tokens that should receive local probe credit.
    """

    trajectory_id: str
    task_id: str
    seed: int
    turn_id: int
    action: str
    action_prefix: Tuple[str, ...]
    state_hash: str
    token_indices: Tuple[int, ...]
    mean_entropy: float
    logprob_margin: float
    invalid_actions_before: int
    legal_actions: Tuple[str, ...]
    max_horizon: int
    suffix_tokens: int
    final_reward: float
    terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def anchor_id(self) -> str:
        return f"{self.trajectory_id}:{self.turn_id}"

    @property
    def invalid_action_ratio(self) -> float:
        return self.invalid_actions_before / max(1, self.turn_id)

    @property
    def turn_fraction(self) -> float:
        return self.turn_id / max(1, self.max_horizon)


@dataclass(frozen=True)
class Anchor:
    """A turn selected by a scheduler for counterfactual probing."""

    turn: TurnRecord
    scheduler: str
    score: float


@dataclass(frozen=True)
class ProbeResult:
    """Measured effect of replacing one action at a fixed replayed state."""

    anchor_id: str
    factual_action: str
    counterfactual_action: str
    factual_reward: float
    counterfactual_reward: float
    delta: float
    additional_rollout_tokens: int
    state_match: bool
    factual_steps: int = 0
    counterfactual_steps: int = 0
    skipped_reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.state_match and self.skipped_reason is None

    @property
    def probe_value(self) -> float:
        if not self.valid or self.additional_rollout_tokens <= 0:
            return 0.0
        return abs(self.delta) / self.additional_rollout_tokens


def as_tuple(values: Sequence[str]) -> Tuple[str, ...]:
    """Convert framework-owned action collections into an immutable public shape."""

    return tuple(str(value) for value in values)

