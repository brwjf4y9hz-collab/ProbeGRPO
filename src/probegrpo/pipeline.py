"""Framework-independent orchestration of scheduler, replay probes, and local credits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Mapping, Sequence

from .advantage import ProbeCredit
from .probing import CounterfactualProber
from .schedulers import AnchorScheduler, choose_alternative_action
from .types import Anchor, ProbeResult, TurnRecord


@dataclass(frozen=True)
class ProbeBatchOutcome:
    anchors: Sequence[Anchor]
    results: Sequence[ProbeResult]
    credits: Sequence[ProbeCredit]
    requested_budget: int

    @property
    def valid_probes(self) -> int:
        return sum(result.valid for result in self.results)

    @property
    def additional_rollout_tokens(self) -> int:
        return sum(result.additional_rollout_tokens for result in self.results)


class ProbePipeline:
    """Execute no more than one configured probe budget for a trajectory group."""

    def __init__(
        self,
        scheduler: AnchorScheduler,
        prober: CounterfactualProber,
        trajectory_to_sample: Mapping[str, int],
        base_seed: int = 0,
    ) -> None:
        self._scheduler = scheduler
        self._prober = prober
        self._trajectory_to_sample = dict(trajectory_to_sample)
        self._base_seed = int(base_seed)

    def run(
        self,
        candidates: Sequence[TurnRecord],
        budget: int,
        training_update: int,
    ) -> ProbeBatchOutcome:
        anchors = self._scheduler.select(candidates, budget, training_update)
        results: List[ProbeResult] = []
        credits: List[ProbeCredit] = []
        for anchor in anchors:
            probe_seed = self._stable_seed(anchor.turn.anchor_id, training_update)
            alternative = choose_alternative_action(anchor.turn, seed=probe_seed)
            result = self._prober.probe(
                anchor,
                factual_action=anchor.turn.action,
                alternative_action=alternative,
                sampling_seed=probe_seed,
            )
            results.append(result)
            self._scheduler.observe(anchor.turn, result)
            if result.valid:
                try:
                    sample_index = self._trajectory_to_sample[anchor.turn.trajectory_id]
                except KeyError as error:
                    raise KeyError(
                        f"No trainer sample index for trajectory {anchor.turn.trajectory_id!r}"
                    ) from error
                credits.append(
                    ProbeCredit(
                        sample_index=sample_index,
                        token_indices=anchor.turn.token_indices,
                        delta=result.delta,
                    )
                )
        return ProbeBatchOutcome(
            anchors=anchors,
            results=results,
            credits=credits,
            requested_budget=budget,
        )

    def _stable_seed(self, anchor_id: str, training_update: int) -> int:
        payload = f"{self._base_seed}:{training_update}:{anchor_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
