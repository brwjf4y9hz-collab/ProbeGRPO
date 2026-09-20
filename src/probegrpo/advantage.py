"""Framework-independent ProbeGRPO advantage shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ProbeCredit:
    """A signed counterfactual delta attached to one sample's assistant-turn tokens."""

    sample_index: int
    token_indices: Sequence[int]
    delta: float
    valid: bool = True


def normalize_probe_deltas(
    deltas: Sequence[float],
    epsilon: float = 1e-8,
) -> List[float]:
    """Bound valid deltas to [-1, 1] without amplifying small differences.

    Use the larger of one reward unit and the largest absolute valid delta as the scale.
    Unlike a centered z-score, a zero reward difference always gives zero credit.
    """

    if not deltas:
        return []
    values = [float(value) for value in deltas]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probe deltas must be finite")
    largest = max(abs(value) for value in values)
    if largest <= epsilon:
        return [0.0] * len(deltas)
    scale = max(1.0, largest)
    return [value / scale for value in values]


def standardize_probe_deltas(
    deltas: Sequence[float],
    epsilon: float = 1e-8,
) -> List[float]:
    """Compatibility alias for the zero-preserving normalization."""

    return normalize_probe_deltas(deltas, epsilon=epsilon)


def blend_probe_advantages(
    base_advantages: Sequence[Sequence[float]],
    credits: Iterable[ProbeCredit],
    lambda_coef: float = 0.5,
) -> List[List[float]]:
    """Add normalized local credit only to explicitly selected token positions."""
    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")
    result = [list(map(float, row)) for row in base_advantages]
    valid_credits = [credit for credit in credits if credit.valid]
    if not valid_credits or lambda_coef == 0:
        return result

    normalized = normalize_probe_deltas([credit.delta for credit in valid_credits])
    for credit, scaled_delta in zip(valid_credits, normalized):
        if not 0 <= credit.sample_index < len(result):
            raise IndexError(f"sample index out of range: {credit.sample_index}")
        row = result[credit.sample_index]
        for token_index in credit.token_indices:
            if not 0 <= token_index < len(row):
                raise IndexError(
                    f"token index {token_index} out of range for sample {credit.sample_index}"
                )
            row[token_index] += lambda_coef * scaled_delta
    return result
