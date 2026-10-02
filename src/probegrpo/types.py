"""本模块集中定义 replay、rollout、turn、anchor 和 probe 的不可变数据结构，约定各组件之间共享的字段含义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReplayState:
    """环境 reset 后或重放一段动作前缀后到达的状态。"""

    observation: str
    state_hash: str
    legal_actions: Tuple[str, ...] = ()
    terminal: bool = False
    reward: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutOutcome:
    """从已重放到的锚点继续执行环境后得到的结果。"""

    reward: float
    token_count: int
    steps: int
    terminal: bool
    actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnRecord:
    """一个助手轮次，以及重放和评分所需的信息。

    ``action_prefix`` contains environment actions strictly before this turn. ``token_indices``
    identifies the assistant tokens that should receive local probe credit.
    """

    trajectory_id: str
    task_id: str
    seed: int
    turn_id: int
    action: str
    action_prefix: Tuple[str, ...]
    state_hash: str
    token_indices: Tuple[int, ...]
    mean_entropy: float
    logprob_margin: float
    invalid_actions_before: int
    legal_actions: Tuple[str, ...]
    max_horizon: int
    suffix_tokens: int
    final_reward: float
    terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def anchor_id(self) -> str:
        return f"{self.trajectory_id}:{self.turn_id}"

    @property
    def invalid_action_ratio(self) -> float:
        return self.invalid_actions_before / max(1, self.turn_id)

    @property
    def turn_fraction(self) -> float:
        return self.turn_id / max(1, self.max_horizon)


@dataclass(frozen=True)
class Anchor:
    """调度器选中用于反事实探测的一个轮次。"""

    turn: TurnRecord
    scheduler: str
    score: float


@dataclass(frozen=True)
class ProbeResult:
    """在固定重放状态下替换一个动作所测得的效果。"""

    anchor_id: str
    factual_action: str
    counterfactual_action: str
    factual_reward: float
    counterfactual_reward: float
    delta: float
    additional_rollout_tokens: int
    state_match: bool
    factual_steps: int = 0
    counterfactual_steps: int = 0
    skipped_reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.state_match and self.skipped_reason is None

    @property
    def probe_value(self) -> float:
        if not self.valid or self.additional_rollout_tokens <= 0:
            return 0.0
        return abs(self.delta) / self.additional_rollout_tokens


def as_tuple(values: Sequence[str]) -> Tuple[str, ...]:
    """将框架管理的动作集合转换为不可变的公共数据形态。"""

    return tuple(str(value) for value in values)
