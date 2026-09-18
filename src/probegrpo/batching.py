"""Pack sparse probe credits into the dense layout expected by verl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .advantage import ProbeCredit, blend_probe_advantages

MaskTensor = Tuple[Tuple[Tuple[bool, ...], ...], ...]
FloatMatrix = Tuple[Tuple[float, ...], ...]
BoolMatrix = Tuple[Tuple[bool, ...], ...]


@dataclass(frozen=True)
class PackedProbeBatch:
    """Dependency-free representation of ``[batch, anchors, tokens]`` probe fields."""

    batch_size: int
    anchors_per_sample: int
    token_count: int
    probe_turn_masks: MaskTensor
    probe_deltas: FloatMatrix
    probe_valid: BoolMatrix

    @property
    def mask_shape(self) -> Tuple[int, int, int]:
        return (self.batch_size, self.anchors_per_sample, self.token_count)

    @property
    def matrix_shape(self) -> Tuple[int, int]:
        return (self.batch_size, self.anchors_per_sample)

    def to_credits(self) -> Tuple[ProbeCredit, ...]:
        """Recover valid sparse credits for dependency-free CPU validation."""

        credits = []
        for sample_index in range(self.batch_size):
            for anchor_index in range(self.anchors_per_sample):
                if not self.probe_valid[sample_index][anchor_index]:
                    continue
                token_indices = tuple(
                    token_index
                    for token_index, selected in enumerate(
                        self.probe_turn_masks[sample_index][anchor_index]
                    )
                    if selected
                )
                credits.append(
                    ProbeCredit(
                        sample_index=sample_index,
                        token_indices=token_indices,
                        delta=self.probe_deltas[sample_index][anchor_index],
                    )
                )
        return tuple(credits)


def pack_probe_credits(
    credits: Sequence[ProbeCredit],
    batch_size: int,
    token_count: int,
    anchors_per_sample: int,
) -> PackedProbeBatch:
    """Pack valid sparse credits without silently truncating malformed input."""

    _validate_dimensions(batch_size, token_count, anchors_per_sample)
    masks: List[List[List[bool]]] = [
        [[False] * token_count for _ in range(anchors_per_sample)]
        for _ in range(batch_size)
    ]
    deltas: List[List[float]] = [
        [0.0] * anchors_per_sample for _ in range(batch_size)
    ]
    valid: List[List[bool]] = [
        [False] * anchors_per_sample for _ in range(batch_size)
    ]
    next_slot = [0] * batch_size
    used_tokens = [set() for _ in range(batch_size)]

    for credit in credits:
        if not credit.valid:
            continue
        sample_index = int(credit.sample_index)
        if not 0 <= sample_index < batch_size:
            raise IndexError(f"sample index out of range: {sample_index}")
        if next_slot[sample_index] >= anchors_per_sample:
            raise ValueError(
                f"sample {sample_index} has more valid credits than anchors_per_sample="
                f"{anchors_per_sample}"
            )
        if not credit.token_indices:
            raise ValueError("a valid probe credit must select at least one token")
        if len(set(credit.token_indices)) != len(credit.token_indices):
            raise ValueError("a probe credit contains duplicate token indices")

        slot = next_slot[sample_index]
        for token_index in credit.token_indices:
            if not 0 <= token_index < token_count:
                raise IndexError(
                    f"token index {token_index} out of range for sample {sample_index}"
                )
            if token_index in used_tokens[sample_index]:
                raise ValueError(
                    f"token index {token_index} is selected by multiple anchors in sample "
                    f"{sample_index}"
                )
            masks[sample_index][slot][token_index] = True
            used_tokens[sample_index].add(token_index)

        deltas[sample_index][slot] = float(credit.delta)
        valid[sample_index][slot] = True
        next_slot[sample_index] += 1

    return PackedProbeBatch(
        batch_size=batch_size,
        anchors_per_sample=anchors_per_sample,
        token_count=token_count,
        probe_turn_masks=_freeze_3d(masks),
        probe_deltas=_freeze_2d_float(deltas),
        probe_valid=_freeze_2d_bool(valid),
    )


def blend_packed_probe_advantages(
    base_advantages: Sequence[Sequence[float]],
    packed: PackedProbeBatch,
    lambda_coef: float = 0.5,
) -> List[List[float]]:
    """Apply a packed batch through the same dependency-free credit implementation."""

    if len(base_advantages) != packed.batch_size:
        raise ValueError("base advantage batch dimension does not match packed probes")
    if any(len(row) != packed.token_count for row in base_advantages):
        raise ValueError("base advantage token dimension does not match packed probes")
    return blend_probe_advantages(base_advantages, packed.to_credits(), lambda_coef)


def _validate_dimensions(batch_size: int, token_count: int, anchors_per_sample: int) -> None:
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if anchors_per_sample < 0:
        raise ValueError("anchors_per_sample must be non-negative")


def _freeze_3d(values: List[List[List[bool]]]) -> MaskTensor:
    return tuple(tuple(tuple(row) for row in sample) for sample in values)


def _freeze_2d_float(values: List[List[float]]) -> FloatMatrix:
    return tuple(tuple(float(value) for value in row) for row in values)


def _freeze_2d_bool(values: List[List[bool]]) -> BoolMatrix:
    return tuple(tuple(bool(value) for value in row) for row in values)
