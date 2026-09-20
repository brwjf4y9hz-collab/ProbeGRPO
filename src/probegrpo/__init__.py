"""ProbeGRPO's framework-independent public API."""

from .advantage import (
    ProbeCredit,
    blend_probe_advantages,
    normalize_probe_deltas,
    standardize_probe_deltas,
)
from .batching import PackedProbeBatch, blend_packed_probe_advantages, pack_probe_credits
from .pipeline import ProbeBatchOutcome, ProbePipeline
from .probing import CounterfactualProber, SuffixPolicy
from .replay import ReplayableEnv, canonical_state_hash
from .rollout import RawAssistantTurn, TrajectoryTrace, extract_turn_records
from .schedulers import EntropyScheduler, LinearUCBScheduler, RandomScheduler
from .types import Anchor, ProbeResult, ReplayState, RolloutOutcome, TurnRecord

__all__ = [
    "Anchor",
    "CounterfactualProber",
    "EntropyScheduler",
    "LinearUCBScheduler",
    "PackedProbeBatch",
    "ProbeCredit",
    "ProbeBatchOutcome",
    "ProbePipeline",
    "ProbeResult",
    "RandomScheduler",
    "RawAssistantTurn",
    "ReplayState",
    "ReplayableEnv",
    "RolloutOutcome",
    "SuffixPolicy",
    "TurnRecord",
    "TrajectoryTrace",
    "blend_probe_advantages",
    "blend_packed_probe_advantages",
    "canonical_state_hash",
    "extract_turn_records",
    "pack_probe_credits",
    "standardize_probe_deltas",
    "normalize_probe_deltas",
]

__version__ = "0.1.0"
