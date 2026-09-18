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


def standardize_probe_deltas(
    deltas: Sequence[float],
    epsilon: float = 1e-8,
) -> List[float]:
    """Standardize deltas without producing NaNs for tiny batches.

    A single probe maps to its sign so the ``budget=1`` ablation remains meaningful. Two or more
    identical deltas map to zero because they contain no relative credit information.
    """

    if not deltas:
        return []
    if len(deltas) == 1:
        value = float(deltas[0])
        return [0.0 if abs(value) <= epsilon else math.copysign(1.0, value)]
    mean = sum(float(value) for value in deltas) / len(deltas)
    variance = sum((float(value) - mean) ** 2 for value in deltas) / len(deltas)
    std = math.sqrt(variance)
    if std <= epsilon:
        return [0.0] * len(deltas)
    return [(float(value) - mean) / std for value in deltas]


def blend_probe_advantages(
    base_advantages: Sequence[Sequence[float]],
    credits: Iterable[ProbeCredit],
    lambda_coef: float = 0.5,
) -> List[List[float]]:
    """Add standardized local credit only to explicitly selected token positions."""
    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")
    result = [list(map(float, row)) for row in base_advantages]
    valid_credits = [credit for credit in credits if credit.valid]
    if not valid_credits or lambda_coef == 0:
        return result

    normalized = standardize_probe_deltas([credit.delta for credit in valid_credits])
    for credit, delta_z in zip(valid_credits, normalized):
        if not 0 <= credit.sample_index < len(result):
            raise IndexError(f"sample index out of range: {credit.sample_index}")
        row = result[credit.sample_index]
        for token_index in credit.token_indices:
            if not 0 <= token_index < len(row):
                raise IndexError(
                    f"token index {token_index} out of range for sample {credit.sample_index}"
                )
            row[token_index] += lambda_coef * delta_z
    return result
