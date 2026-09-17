"""Paired factual/counterfactual suffix evaluation."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

from .replay import ReplayableEnv
from .types import Anchor, ProbeResult, ReplayState, RolloutOutcome


class SuffixPolicy(Protocol):
    """Continue from an anchor after forcing the first action of the suffix."""

    def __call__(
        self,
        env: ReplayableEnv,
        anchor_state: ReplayState,
        forced_action: str,
        sampling_seed: int,
    ) -> RolloutOutcome:
        ...


class CounterfactualProber:
    """Measure a turn's signed reward effect under exact prefix replay."""

    def __init__(
        self,
        env_factory: Callable[[], ReplayableEnv],
        suffix_policy: SuffixPolicy,
        state_hash_strict: bool = True,
    ) -> None:
        self._env_factory = env_factory
        self._suffix_policy = suffix_policy
        self._state_hash_strict = state_hash_strict

    def probe(
        self,
        anchor: Anchor,
        factual_action: Optional[str],
        alternative_action: Optional[str],
        sampling_seed: int,
    ) -> ProbeResult:
        turn = anchor.turn
        factual = factual_action or turn.action
        alternative = alternative_action or ""
        if not alternative:
            return self._skipped(anchor, factual, alternative, "no_alternative_action")
        if factual == alternative:
            return self._skipped(anchor, factual, alternative, "actions_are_identical")
        if factual not in turn.legal_actions:
            return self._skipped(anchor, factual, alternative, "factual_action_is_not_legal")
        if alternative not in turn.legal_actions:
            return self._skipped(anchor, factual, alternative, "alternative_action_is_not_legal")

        factual_env = self._env_factory()
        counterfactual_env = self._env_factory()
        try:
            factual_state = factual_env.replay(turn.task_id, turn.seed, turn.action_prefix)
            counterfactual_state = counterfactual_env.replay(
                turn.task_id, turn.seed, turn.action_prefix
            )
        except Exception as error:  # adapters can raise framework-specific replay errors
            return self._skipped(
                anchor,
                factual,
                alternative,
                f"replay_failed:{type(error).__name__}:{error}",
            )

        states_match = (
            factual_state.state_hash == counterfactual_state.state_hash == turn.state_hash
            and factual_state.observation == counterfactual_state.observation
        )
        if self._state_hash_strict and not states_match:
            return self._skipped(anchor, factual, alternative, "anchor_state_mismatch")

        try:
            factual_outcome = self._suffix_policy(
                factual_env, factual_state, factual, sampling_seed
            )
            counterfactual_outcome = self._suffix_policy(
                counterfactual_env, counterfactual_state, alternative, sampling_seed
            )
        except Exception as error:
            return self._skipped(
                anchor,
                factual,
                alternative,
                f"suffix_failed:{type(error).__name__}:{error}",
            )

        return ProbeResult(
            anchor_id=turn.anchor_id,
            factual_action=factual,
            counterfactual_action=alternative,
            factual_reward=float(factual_outcome.reward),
            counterfactual_reward=float(counterfactual_outcome.reward),
            delta=float(factual_outcome.reward - counterfactual_outcome.reward),
            additional_rollout_tokens=(
                int(factual_outcome.token_count) + int(counterfactual_outcome.token_count)
            ),
            state_match=states_match,
            factual_steps=factual_outcome.steps,
            counterfactual_steps=counterfactual_outcome.steps,
        )

    @staticmethod
    def _skipped(
        anchor: Anchor,
        factual_action: str,
        alternative_action: str,
        reason: str,
    ) -> ProbeResult:
        return ProbeResult(
            anchor_id=anchor.turn.anchor_id,
            factual_action=factual_action,
            counterfactual_action=alternative_action,
            factual_reward=0.0,
            counterfactual_reward=0.0,
            delta=0.0,
            additional_rollout_tokens=0,
            state_match=False,
            skipped_reason=reason,
        )

