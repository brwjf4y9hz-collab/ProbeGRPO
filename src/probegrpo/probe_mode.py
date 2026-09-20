"""Decide whether one AgentLoop session should spend rollout tokens on a probe.

The current trainer gate supports one random anchor in session zero of a prompt
group. Unsupported budgets or schedulers fail early instead of silently running
fewer probes than requested or mislabeling an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProbeMode:
    run_probe: bool
    credit_requested: bool
    scheduler: str


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
        if isinstance(budget, bool) or not isinstance(budget, int) or budget not in (0, 1):
            raise ValueError("this trainer gate supports probe.budget=0 or 1 only")
        scheduler = settings.get("scheduler", "random")
        if scheduler != "random":
            raise ValueError("this trainer gate supports probe.scheduler=random only")
        return ProbeMode(
            run_probe=budget == 1 and session_id == 0,
            credit_requested=budget == 1,
            scheduler="random",
        )
    return ProbeMode(
        run_probe=debug_probe and session_id == 0,
        credit_requested=False,
        scheduler="random_debug" if debug_probe else "none",
    )
