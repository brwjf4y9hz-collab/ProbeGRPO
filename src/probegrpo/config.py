"""Validated configuration for ProbeGRPO-specific behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeConfig:
    budget: int = 2
    lambda_coef: float = 0.5
    scheduler: str = "linear_ucb"
    warmup_updates: int = 20
    warmup_probes: int = 200
    exploration_rate: float = 0.1
    linear_ucb_alpha: float = 1.0
    state_hash_strict: bool = True

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if self.lambda_coef < 0:
            raise ValueError("lambda_coef must be non-negative")
        if self.scheduler not in {"random", "entropy", "linear_ucb"}:
            raise ValueError(f"unsupported scheduler: {self.scheduler}")
        if self.warmup_updates < 0 or self.warmup_probes < 0:
            raise ValueError("warm-up values must be non-negative")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("exploration_rate must be in [0, 1]")


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
STRETCH_MODEL = "Qwen/Qwen3.5-4B"

