"""本模块实现固定预算下的锚点选择策略，包括随机、chosen-token surprisal 和 LinearUCB；所有策略都从可观测 turn 特征中选点。"""
# 一条 trajectory 含有多个 turn；本模块决定在固定额外 rollout 预算内选择哪些 turn 做反事实 probe。
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

from .types import Anchor, ProbeResult, TurnRecord

FEATURE_DIM = 7


def turn_features(turn: TurnRecord) -> Tuple[float, ...]:
    """生成有界且可解释的在线 probe 价值预测特征。

    特征仅依赖 rollout 时记录的状态、轮次和 chosen-token surprisal，不读取
    尚不可得的完整策略分布 entropy。LinearUCB 使用固定长度向量进行打分。
    """
    entropy = _clip(turn.mean_entropy / 10.0)
    uncertainty = 1.0 / (1.0 + max(0.0, turn.logprob_margin))
    position = _clip(turn.turn_fraction)
    invalid_ratio = _clip(turn.invalid_action_ratio)
    legal_count = _clip(math.log1p(len(turn.legal_actions)) / math.log(33.0))
    suffix_cost = _clip(math.log1p(max(0, turn.suffix_tokens)) / math.log(8193.0))
    outcome = max(-1.0, min(1.0, float(turn.final_reward)))
    return (entropy, ##模型有多犹豫。
            uncertainty, ##第一候选 action和第二候选 action之间差多少
            position, ##表示这个 turn 在 trajectory 的什么位置
            invalid_ratio, ##Agent 最近/当前产生非法 action 的比例
            legal_count, ##当前有多少种可选 action
            suffix_cost, ##probe cost 特征
            outcome)##这条 trajectory 最终结果怎么样


def _clip(value: float) -> float:##这样 Linear UCB 数值更稳定。
    return max(0.0, min(1.0, float(value)))


def _eligible(candidates: Iterable[TurnRecord]) -> List[TurnRecord]:##先过滤哪些 turn 能 probe
    return [
        turn
        for turn in candidates
        if not turn.terminal
        and len(set(turn.legal_actions)) >= 2
        and turn.action in turn.legal_actions
    ]


class AnchorScheduler(ABC):##这是所有 scheduler 的抽象基类
    """在不改变 rollout 预算的前提下选择固定数量的 turn。"""

    name = "base"

    @abstractmethod
    def select(
        self,
        candidates: Sequence[TurnRecord],
        budget: int,
        training_update: int = 0,
    ) -> List[Anchor]:
        raise NotImplementedError

    def observe(self, turn: TurnRecord, result: ProbeResult) -> None:
        """可选地根据已选锚点测得的价值更新调度器。"""
        return None


class RandomScheduler(AnchorScheduler):#不做聪明选择，随机 probe
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(
        self,
        candidates: Sequence[TurnRecord],
        budget: int,
        training_update: int = 0,
    ) -> List[Anchor]:
        del training_update
        eligible = _eligible(candidates)
        if budget <= 0 or not eligible:
            return []
        chosen = self._rng.sample(eligible, k=min(budget, len(eligible)))
        return [Anchor(turn=turn, scheduler=self.name, score=0.0) for turn in chosen]


class EntropyScheduler(AnchorScheduler):##越不确定的 turn，越值得花 probe budget 去看
    name = "entropy"

    def select(
        self,
        candidates: Sequence[TurnRecord],
        budget: int,
        training_update: int = 0,
    ) -> List[Anchor]:
        del training_update
        if budget <= 0:
            return []
        ranked = sorted(
            _eligible(candidates),
            key=lambda turn: (-turn.mean_entropy, turn.anchor_id),
        )
        return [
            Anchor(turn=turn, scheduler=self.name, score=turn.mean_entropy)
            for turn in ranked[:budget]
        ]


class LinearUCBScheduler(AnchorScheduler):##边训练边学习：什么样的 turn 最值得 probe
    """使用 Sherman-Morrison 更新的在线成本感知锚点调度器。
    The regression target is ``abs(delta_reward) / additional_rollout_tokens``. During warm-up,
    anchors are sampled across early/middle/late trajectory thirds. After warm-up, LinearUCB ranks
    candidates, while ``exploration_rate`` reserves occasional random batches.
    每次观测到 probe 结果后更新线性模型的逆协方差近似；分数中同时考虑
    预测价值、不确定性探索项和估计 token 成本。此策略属于启发式调度器，
    不提供无偏因果估计保证。
    """

    name = "linear_ucb"

    def __init__(
        self,
        alpha: float = 1.0,
        warmup_updates: int = 20,##warmup
        warmup_probes: int = 200,
        exploration_rate: float = 0.1,
        l2: float = 1.0,
        seed: int = 0,
    ) -> None:
        if alpha < 0 or l2 <= 0:
            raise ValueError("alpha must be non-negative and l2 must be positive")
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        self.alpha = float(alpha)
        self.warmup_updates = int(warmup_updates)
        self.warmup_probes = int(warmup_probes)
        self.exploration_rate = float(exploration_rate)
        self._rng = random.Random(seed)
        self._a_inv = [
            [1.0 / l2 if row == col else 0.0 for col in range(FEATURE_DIM)]
            for row in range(FEATURE_DIM)
        ]
        self._b = [0.0] * FEATURE_DIM
        self.observations = 0

    @property
    def theta(self) -> Tuple[float, ...]:
        return tuple(_mat_vec(self._a_inv, self._b))

    def select(
        self,
        candidates: Sequence[TurnRecord],
        budget: int,
        training_update: int = 0,
    ) -> List[Anchor]:
        eligible = _eligible(candidates)
        if budget <= 0 or not eligible:
            return []
        count = min(budget, len(eligible))
        warming_up = (
            training_update < self.warmup_updates or self.observations < self.warmup_probes
        )
        if warming_up:
            selected = self._stratified_sample(eligible, count)
            return [
                Anchor(turn=turn, scheduler="linear_ucb_warmup", score=0.0)
                for turn in selected
            ]
        if self._rng.random() < self.exploration_rate:##保留 10% random exploration
            selected = self._rng.sample(eligible, count)
            return [
                Anchor(turn=turn, scheduler="linear_ucb_explore", score=0.0)
                for turn in selected
            ]

        ranked = sorted(##用 LinearUCB 排名
            ((self.score(turn), turn) for turn in eligible),
            key=lambda pair: (-pair[0], pair[1].anchor_id),
        )
        return [
            Anchor(turn=turn, scheduler=self.name, score=score)
            for score, turn in ranked[:count]
        ]

    def score(self, turn: TurnRecord) -> float:
        x = list(turn_features(turn))
        theta = list(self.theta)
        mean = _dot(theta, x)
        uncertainty = math.sqrt(max(0.0, _dot(x, _mat_vec(self._a_inv, x))))
        return mean + self.alpha * uncertainty

    def observe(self, turn: TurnRecord, result: ProbeResult) -> None:##更新 LinearUCB
        if not result.valid or result.additional_rollout_tokens <= 0:
            return
        x = list(turn_features(turn))
        a_inv_x = _mat_vec(self._a_inv, x)
        denominator = 1.0 + _dot(x, a_inv_x)
        outer = [
            [a_inv_x[row] * a_inv_x[col] / denominator for col in range(FEATURE_DIM)]
            for row in range(FEATURE_DIM)
        ]
        self._a_inv = [
            [self._a_inv[row][col] - outer[row][col] for col in range(FEATURE_DIM)]
            for row in range(FEATURE_DIM)
        ]
        target = result.probe_value
        self._b = [value + target * feature for value, feature in zip(self._b, x)]
        self.observations += 1

    def _stratified_sample(self, eligible: Sequence[TurnRecord], count: int) -> List[TurnRecord]:
        # warmup 阶段按位置分层抽样，避免样本只落在轨迹开头或结尾。
        buckets = {0: [], 1: [], 2: []}
        for turn in eligible:
            bucket = min(2, int(_clip(turn.turn_fraction) * 3))
            buckets[bucket].append(turn)
        for values in buckets.values():
            self._rng.shuffle(values)

        selected: List[TurnRecord] = []
        while len(selected) < count and any(buckets.values()):
            for bucket in (0, 1, 2):
                if buckets[bucket] and len(selected) < count:
                    selected.append(buckets[bucket].pop())
        return selected


def choose_alternative_action(turn: TurnRecord, seed: Optional[int] = None) -> Optional[str]:
    """以可复现方式选择一个与事实动作不同的合法动作。"""

    alternatives = sorted(set(turn.legal_actions) - {turn.action})
    if not alternatives:
        return None
    rng = random.Random(seed)
    return alternatives[rng.randrange(len(alternatives))]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [_dot(row, vector) for row in matrix]
