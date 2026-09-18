"""Reusable suffix policies for real model callbacks and deterministic testing."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from .replay import ReplayableEnv
from .types import ReplayState, RolloutOutcome

CompletionFn = Callable[[str, Sequence[str], Tuple[str, ...], int], str]
TokenCounter = Callable[[str], int]


class CallbackSuffixPolicy:
    """Continue a replay branch using a vLLM/OpenAI-compatible callback.

    The callback receives `(observation, legal_actions, prior_actions, sampling_seed)` and must
    return one serialized environment action. This keeps serving concerns outside the core package.
    """

    def __init__(
        self,
        completion_fn: CompletionFn,
        max_steps: int,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._completion_fn = completion_fn
        self._max_steps = int(max_steps)
        self._token_counter = token_counter or (lambda text: max(1, len(text.split())))

    def __call__(
        self,
        env: ReplayableEnv,
        anchor_state: ReplayState,
        forced_action: str,
        sampling_seed: int,
    ) -> RolloutOutcome:
        state = env.step(forced_action)
        actions: List[str] = [forced_action]
        token_count = self._token_counter(forced_action)
        while not state.terminal and len(actions) < self._max_steps:
            action = self._completion_fn(
                state.observation,
                state.legal_actions,
                tuple(actions),
                sampling_seed + len(actions),
            )
            token_count += self._token_counter(action)
            actions.append(action)
            state = env.step(action)
        return RolloutOutcome(
            reward=state.reward,
            token_count=token_count,
            steps=len(actions),
            terminal=state.terminal,
            actions=tuple(actions),
        )
