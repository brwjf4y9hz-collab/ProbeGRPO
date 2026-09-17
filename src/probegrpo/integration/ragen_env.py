"""Replay adapter for RAGEN environments without importing RAGEN at package import time."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ..replay import ReplayableEnv, canonical_state_hash
from ..types import ReplayState


class RagenReplayableEnv(ReplayableEnv):
    """Wrap a fresh RAGEN environment instance in ProbeGRPO's replay contract.

    RAGEN environments are task-seeded, so ``task_id`` is metadata while ``seed`` determines the
    actual instance. ``action_encoder`` converts serialized actions back to the native environment
    type; it defaults to integer conversion for digit strings and identity otherwise.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        mode: str = "train",
        action_encoder: Optional[Callable[[str], Any]] = None,
        legal_action_provider: Optional[Callable[[Any], Sequence[Any]]] = None,
    ) -> None:
        self._env = env_factory()
        self._mode = mode
        self._action_encoder = action_encoder
        self._legal_action_provider = legal_action_provider
        self._task_id = ""
        self._seed = 0
        self._terminal = False
        self._reward = 0.0
        self._observation = ""
        self._step_index = 0

    @property
    def raw_env(self) -> Any:
        return self._env

    def reset(self, task_id: str, seed: int) -> ReplayState:
        self._task_id = task_id
        self._seed = int(seed)
        self._terminal = False
        self._reward = 0.0
        self._step_index = 0
        try:
            observation = self._env.reset(seed=seed, mode=self._mode)
        except TypeError:
            observation = self._env.reset(seed=seed)
        self._observation = str(observation)
        return self._state()

    def step(self, action: str) -> ReplayState:
        if self._terminal:
            raise RuntimeError("Cannot step a terminal RAGEN environment")
        native_action = (
            self._action_encoder(action)
            if self._action_encoder is not None
            else _encode_for_env(self._env, action)
        )
        observation, reward, done, info = self._env.step(native_action)
        self._observation = str(observation)
        self._reward = float(reward)
        self._terminal = bool(done)
        self._step_index += 1
        return self._state(info=info)

    def _state(self, info: Optional[dict] = None) -> ReplayState:
        legal_actions = tuple(str(action) for action in self._legal_actions())
        hash_payload = {
            "task_id": self._task_id,
            "seed": self._seed,
            "step": self._step_index,
            "observation": self._observation,
            "legal_actions": legal_actions,
            "terminal": self._terminal,
        }
        return ReplayState(
            observation=self._observation,
            state_hash=canonical_state_hash(hash_payload),
            legal_actions=legal_actions,
            terminal=self._terminal,
            reward=self._reward,
            metadata={"info": dict(info or {}), "step": self._step_index},
        )

    def _legal_actions(self) -> Sequence[Any]:
        if self._terminal:
            return ()
        if self._legal_action_provider is not None:
            return self._legal_action_provider(self._env)
        if hasattr(self._env, "get_available_actions"):
            return self._env.get_available_actions()
        if hasattr(self._env, "get_all_actions"):
            actions = self._env.get_all_actions()
            lookup = getattr(self._env, "ACTION_LOOKUP", {})
            if lookup:
                return [lookup.get(action, action) for action in actions]
            return actions
        raise TypeError("RAGEN environment does not expose a legal-action method")


def make_ragen_action_encoder(env: Any) -> Callable[[str], Any]:
    """Build an encoder that accepts either RAGEN action names or numeric IDs."""

    lookup = getattr(env, "ACTION_LOOKUP", {})
    reverse = {str(value).lower(): key for key, value in lookup.items()}

    def encode(action: str) -> Any:
        normalized = str(action).strip().lower()
        if normalized in reverse:
            return reverse[normalized]
        return _default_action_encoder(action)

    return encode


def _encode_for_env(env: Any, action: str) -> Any:
    lookup = getattr(env, "ACTION_LOOKUP", {})
    normalized = str(action).strip().lower()
    for action_id, action_name in lookup.items():
        if str(action_name).strip().lower() == normalized:
            return action_id
    return _default_action_encoder(action)


def _default_action_encoder(action: str) -> Any:
    normalized = str(action).strip()
    return int(normalized) if normalized.lstrip("-").isdigit() else normalized
