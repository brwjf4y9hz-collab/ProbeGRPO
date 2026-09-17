"""Small, dependency-free experiment record and aggregation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class ExperimentRecord:
    method: str
    task: str
    seed: int
    update: int
    reward: float
    success: float
    episode_steps: float
    invalid_action_rate: float
    main_rollout_tokens: int
    probe_rollout_tokens: int
    wall_clock_seconds: float
    gpu_hours: float = 0.0
    high_impact_anchors: int = 0

    @property
    def total_rollout_tokens(self) -> int:
        return self.main_rollout_tokens + self.probe_rollout_tokens

    @property
    def probe_cost_ratio(self) -> float:
        return self.probe_rollout_tokens / max(1, self.main_rollout_tokens)

    @property
    def high_impact_per_1k_probe_tokens(self) -> float:
        return 1000.0 * self.high_impact_anchors / max(1, self.probe_rollout_tokens)


def append_jsonl(path: Path, record: ExperimentRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[ExperimentRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ExperimentRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid record at {path}:{line_number}: {error}") from error
    return records


def summarize_records(
    records: Iterable[ExperimentRecord],
) -> Dict[Tuple[str, str], Mapping[str, float]]:
    groups: Dict[Tuple[str, str], List[ExperimentRecord]] = {}
    for record in records:
        groups.setdefault((record.method, record.task), []).append(record)

    summaries: Dict[Tuple[str, str], Mapping[str, float]] = {}
    for key, values in sorted(groups.items()):
        rewards = [value.reward for value in values]
        successes = [value.success for value in values]
        summaries[key] = {
            "runs": float(len(values)),
            "reward_mean": mean(rewards),
            "reward_std": _population_std(rewards),
            "success_mean": mean(successes),
            "success_std": _population_std(successes),
            "total_rollout_tokens": float(sum(value.total_rollout_tokens for value in values)),
            "probe_cost_ratio_mean": mean(value.probe_cost_ratio for value in values),
            "gpu_hours": sum(value.gpu_hours for value in values),
            "wall_clock_seconds": sum(value.wall_clock_seconds for value in values),
            "high_impact_per_1k_probe_tokens": mean(
                value.high_impact_per_1k_probe_tokens for value in values
            ),
        }
    return summaries


def _population_std(values: List[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return math.sqrt(mean((value - center) ** 2 for value in values))

