"""本模块定义可回放环境协议、规范化状态哈希和动作前缀重放。哈希用于检测状态不一致，不单独证明环境隐藏状态或外部副作用完全一致。"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .types import ReplayState


def canonical_state_hash(state: Any) -> str:
    """为 JSON 兼容的环境状态生成确定性的 SHA-256 哈希。"""

    payload = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"State value is not JSON serializable: {type(value)!r}")


class ReplayableEnv(ABC):
    """精确反事实前缀回放所需的最小环境协议。

    Implementations must make ``reset(task_id, seed)`` deterministic. The default ``replay``
    method intentionally re-executes public actions instead of relying on unsafe object copies.
    """

    @abstractmethod
    def reset(self, task_id: str, seed: int) -> ReplayState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> ReplayState:
        raise NotImplementedError

    def replay(self, task_id: str, seed: int, action_prefix: Sequence[str]) -> ReplayState:
        state = self.reset(task_id=task_id, seed=seed)
        for index, action in enumerate(action_prefix):
            if state.terminal:
                raise ReplayError(
                    f"Prefix contains action {index} after terminal state for task {task_id!r}"
                )
            state = self.step(action)
        return state


class ReplayError(RuntimeError):
    """动作前缀无法安全重放时抛出的异常。"""


def state_from_mapping(
    observation: str,
    raw_state: Mapping[str, Any],
    legal_actions: Sequence[str] = (),
    terminal: bool = False,
    reward: float = 0.0,
) -> ReplayState:
    """供第三方环境适配器使用的便捷状态构造函数。"""

    return ReplayState(
        observation=observation,
        state_hash=canonical_state_hash(raw_state),
        legal_actions=tuple(legal_actions),
        terminal=terminal,
        reward=float(reward),
        metadata={"raw_state": dict(raw_state)},
    )
