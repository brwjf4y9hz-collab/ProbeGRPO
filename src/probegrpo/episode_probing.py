"""Replay an agent episode into paired, model-generated suffixes.

The recorded prefix and anchor action are injected without another model call. Both
branches then use fresh generation with the same sampling seed. This module has no
torch or verl dependency; the runtime supplies token I/O and an environment factory.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from .agent_episode import Episode, GeneratedAction, run_episode
from .replay import ReplayableEnv
from .types import ProbeResult


async def probe_episode(
    episode: Episode,
    anchor_turn_id: int,
    alternative: GeneratedAction,
    *,
    env_factory: Callable[[], ReplayableEnv],
    io_factory: Callable[[int, str], object],
    sampling_seed: int,
    response_budget: int,
    timeout_seconds: Optional[float] = None,
) -> ProbeResult:
    """Replace one legal action, then resample both suffixes from equal contexts.

    ``io_factory(seed, branch_name)`` must create independent model sessions. The
    seed is identical in the factual and counterfactual calls. Invalid probes
    return zero credit and a reason instead of changing a training advantage.
    """

    if not 0 <= anchor_turn_id < len(episode.turns):
        raise IndexError("anchor_turn_id is outside the episode")
    turn = episode.turns[anchor_turn_id]
    anchor_id = f"{episode.trajectory_id}:{anchor_turn_id}"
    factual_action = turn.action
    alternative_action = alternative.text.strip().lower()

    def skipped(reason: str) -> ProbeResult:
        return ProbeResult(
            anchor_id=anchor_id,
            factual_action=factual_action,
            counterfactual_action=alternative_action,
            factual_reward=0.0,
            counterfactual_reward=0.0,
            delta=0.0,
            additional_rollout_tokens=0,
            state_match=False,
            skipped_reason=reason,
        )

    if factual_action not in turn.legal_actions:
        return skipped("factual_action_is_not_legal")
    if alternative_action == factual_action:
        return skipped("actions_are_identical")
    if alternative_action not in turn.legal_actions:
        return skipped("alternative_action_is_not_legal")
    if not alternative.token_ids:
        return skipped("alternative_has_no_tokens")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    try:
        factual_env, counterfactual_env = env_factory(), env_factory()
        factual_state = factual_env.replay(
            episode.task_id, episode.seed, turn.action_prefix
        )
        counterfactual_state = counterfactual_env.replay(
            episode.task_id, episode.seed, turn.action_prefix
        )
    except Exception as error:
        return skipped(f"replay_failed:{type(error).__name__}:{error}")
    if not (
        factual_state.state_hash
        == counterfactual_state.state_hash
        == turn.state_hash
        and factual_state.observation
        == counterfactual_state.observation
        == turn.observation
        and factual_state.legal_actions == counterfactual_state.legal_actions
    ):
        return skipped("anchor_state_mismatch")

    scripted_prefix = {}
    for index, recorded in enumerate(episode.turns[:anchor_turn_id]):
        if not recorded.generated_token_ids:
            return skipped("missing_recorded_prefix_tokens")
        scripted_prefix[index] = GeneratedAction(
            recorded.generated_token_ids,
            recorded.raw_text,
            recorded.generated_logprobs,
        )
    if not turn.generated_token_ids:
        return skipped("missing_recorded_anchor_tokens")

    async def branch(name: str, forced: GeneratedAction) -> Episode:
        scripted = {
            **scripted_prefix,
            anchor_turn_id: forced,
        }
        result = run_episode(
            io_factory(sampling_seed, name),
            env_factory(),
            task_id=episode.task_id,
            seed=episode.seed,
            trajectory_id=f"{episode.trajectory_id}:{name}",
            max_turns=episode.max_horizon,
            response_budget=response_budget,
            scripted_actions=scripted,
        )
        if timeout_seconds is not None:
            return await asyncio.wait_for(result, timeout=timeout_seconds)
        return await result

    factual = GeneratedAction(
        turn.generated_token_ids, turn.raw_text, turn.generated_logprobs
    )
    try:
        factual_episode = await branch("factual", factual)
        counterfactual_episode = await branch("counterfactual", alternative)
    except Exception as error:
        return skipped(f"suffix_failed:{type(error).__name__}:{error}")

    if not (
        _same_anchor_context(episode, factual_episode, anchor_turn_id)
        and _same_anchor_context(episode, counterfactual_episode, anchor_turn_id)
    ):
        return skipped("anchor_context_mismatch")

    factual_tokens = _suffix_tokens(factual_episode, anchor_turn_id)
    counterfactual_tokens = _suffix_tokens(counterfactual_episode, anchor_turn_id)
    return ProbeResult(
        anchor_id=anchor_id,
        factual_action=factual_action,
        counterfactual_action=alternative_action,
        factual_reward=factual_episode.final_reward,
        counterfactual_reward=counterfactual_episode.final_reward,
        delta=factual_episode.final_reward - counterfactual_episode.final_reward,
        additional_rollout_tokens=factual_tokens + counterfactual_tokens,
        state_match=True,
        factual_steps=len(factual_episode.turns) - anchor_turn_id,
        counterfactual_steps=len(counterfactual_episode.turns) - anchor_turn_id,
    )


def _same_anchor_context(original: Episode, branch: Episode, anchor_turn_id: int) -> bool:
    if len(branch.turns) <= anchor_turn_id:
        return False
    expected, actual = original.turns[anchor_turn_id], branch.turns[anchor_turn_id]
    if (
        expected.state_hash != actual.state_hash
        or expected.observation != actual.observation
        or expected.action_prefix != actual.action_prefix
    ):
        return False
    expected_end = original.stream.prompt_length + expected.token_indices[0]
    actual_end = branch.stream.prompt_length + actual.token_indices[0]
    return (
        original.stream.token_ids[:expected_end] == branch.stream.token_ids[:actual_end]
        and original.stream.response_mask[: expected.token_indices[0]]
        == branch.stream.response_mask[: actual.token_indices[0]]
    )


def _suffix_tokens(episode: Episode, anchor_turn_id: int) -> int:
    first_token = episode.turns[anchor_turn_id].token_indices[0]
    return len(episode.stream.response_mask) - first_token
