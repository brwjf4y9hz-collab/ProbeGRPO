"""Decide whether one AgentLoop session should spend rollout tokens on a probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProbeMode:
    run_probe: bool
    credit_requested: bool
    scheduler: str
    group_budget: int = 0


def resolve_probe_mode(
    settings: Mapping[str, Any],
    *,
    session_id: int,
    debug_probe: bool = False,
) -> ProbeMode:
    """Resolve training config first; the legacy debug switch is fallback only."""

    if session_id < 0:
        raise ValueError("session_id must be non-negative")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("probe.enabled must be a boolean")
    if enabled:
        budget = settings.get("budget", 1)
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 4:
            raise ValueError("probe.budget must be an integer in [0, 4]")
        scheduler = settings.get("scheduler", "random")
        if scheduler not in {"random", "surprisal", "linear_ucb"}:
            raise ValueError("probe.scheduler must be random, surprisal, or linear_ucb")
        return ProbeMode(
            run_probe=session_id < budget,
            credit_requested=budget > 0,
            scheduler=scheduler,
            group_budget=budget,
        )
    return ProbeMode(
        run_probe=debug_probe and session_id == 0,
        credit_requested=False,
        scheduler="random_debug" if debug_probe else "none",
        group_budget=1 if debug_probe else 0,
    )
