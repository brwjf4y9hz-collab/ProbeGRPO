"""本模块把反事实奖励差转换为稀疏的 token 级局部优势。归一化保持零差值为零；只修改明确标记的助手动作 token，输入矩阵不会被原地改写。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ProbeCredit:
    """附加到某个样本助手动作 token 上的有符号反事实奖励差。"""

    sample_index: int
    token_indices: Sequence[int]
    delta: float
    valid: bool = True


def normalize_probe_deltas(
    deltas: Sequence[float],
    epsilon: float = 1e-8,
) -> List[float]:
    """将奖励差缩放到 ``[-1, 1]``，且不放大小幅差异。

    缩放因子取 1 个奖励单位与有效差值最大绝对值中的较大者。该方式不做
    去均值标准化，因此零差值仍为零，也不会因为同批其他样本而变成非零。
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
    """保留旧接口名称，实际调用保持零值不变的归一化函数。"""

    return normalize_probe_deltas(deltas, epsilon=epsilon)


def blend_probe_advantages(
    base_advantages: Sequence[Sequence[float]],
    credits: Iterable[ProbeCredit],
    lambda_coef: float = 0.5,
) -> List[List[float]]:
    """只在显式指定的 token 位置加入归一化局部 credit。

    每个有效 probe 的同一个缩放奖励差会加到其 token_indices；调用方负责
    确保这些坐标只指向目标助手动作 token。函数会检查样本和 token 下标，
    并返回新矩阵，避免原地修改传入的 GRPO 优势。
    """
    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")
    result = [list(map(float, row)) for row in base_advantages]
    valid_credits = [credit for credit in credits if credit.valid]
    if not valid_credits or lambda_coef == 0:
        return result

    # 在同一批有效 probe 上使用同一缩放因子，随后逐个写入其动作 token。
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
