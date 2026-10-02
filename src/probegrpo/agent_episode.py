"""本模块驱动一次完整的多轮 Agent episode，并记录对齐训练 response 的 token 流、mask、动作前状态和回放前缀。环境与模型 token I/O 通过接口注入，便于 CPU 验证和真实运行时复用。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Tuple

from .envs.sokoban import ACTIONS, initial_messages, observation_message
from .rollout import RawAssistantTurn, TrajectoryTrace


@dataclass(frozen=True)
class TokenStream:
    """保存完整运行时 token 序列，并提供仅针对 response 的坐标元数据。"""

    token_ids: Tuple[int, ...]
    response_mask: Tuple[int, ...] = ()
    logprobs: Optional[Tuple[float, ...]] = None

    @property
    def prompt_length(self):
        return len(self.token_ids) - len(self.response_mask)

    def validate(self):
        if self.prompt_length <= 0 or any(x not in (0, 1) for x in self.response_mask):
            raise ValueError("Invalid prompt length or response mask")
        if self.logprobs is not None:
            if len(self.logprobs) != len(self.response_mask):
                raise ValueError("Logprobs must align with response coordinates")
            if not all(math.isfinite(x) for x in self.logprobs):
                raise ValueError("Non-finite rollout logprob")


def validate_merge(previous: TokenStream, updated: TokenStream, *, assistant: bool):
    """若 tokenizer 改写了已记录的助手 token，则拒绝继续。

    Untrained trailing context may be rewritten by Continuous Token boundaries. Earlier
    trained tokens and their response coordinates must remain exactly unchanged.
    """
    updated.validate()
    if updated.prompt_length != previous.prompt_length:
        raise ValueError("Token merge changed initial prompt boundary")
    trained = [i for i, flag in enumerate(previous.response_mask) if flag]
    end = max(trained, default=-1) + 1
    absolute_end = previous.prompt_length + end
    if (
        updated.token_ids[:absolute_end] != previous.token_ids[:absolute_end]
        or updated.response_mask[:end] != previous.response_mask[:end]
    ):
        raise ValueError("Token merge rewrote previously recorded assistant tokens")
    added = tuple(i for i in range(end, len(updated.response_mask)) if updated.response_mask[i])
    if not assistant and added:
        raise ValueError("Observation tokens cannot receive assistant credit")
    return added


@dataclass(frozen=True)
class GeneratedAction:
    token_ids: Tuple[int, ...]
    text: str
    logprobs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class EpisodeTurn:
    action: str
    raw_text: str
    action_prefix: Tuple[str, ...]
    observation: str
    state_hash: str
    next_observation: str
    next_state_hash: str
    legal_actions: Tuple[str, ...]
    action_valid: bool
    token_indices: Tuple[int, ...]
    reward: float
    done_after: bool
    generated_token_ids: Tuple[int, ...] = ()
    generated_logprobs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class Episode:
    trajectory_id: str
    task_id: str
    seed: int
    max_horizon: int
    stream: TokenStream
    turns: Tuple[EpisodeTurn, ...]
    final_reward: float
    stop_reason: str

    def as_dict(self):
        return {
            **asdict(self),
            "token_coordinate_system": "response",
            "token_entropy": None,
            "top2_margin": None,
            "statistics_status": "requires_actor_statistics",
        }

    def to_trace(self, *, entropies, top1_logprobs, top2_logprobs) -> TrajectoryTrace:
        """在 actor 侧统计量可用后建立数据桥接；不得从 log p(a) 臆造 entropy。"""
        count = len(self.stream.response_mask)
        for values in (entropies, top1_logprobs, top2_logprobs):
            if len(values) != count or not all(math.isfinite(x) for x in values):
                raise ValueError("Actor statistics must be finite and response-aligned")
        return TrajectoryTrace(
            trajectory_id=self.trajectory_id,
            task_id=self.task_id,
            seed=self.seed,
            token_count=count,
            max_horizon=self.max_horizon,
            final_reward=self.final_reward,
            turns=tuple(
                RawAssistantTurn(
                    action=t.action,
                    state_hash=t.state_hash,
                    token_indices=t.token_indices,
                    token_entropies=tuple(entropies[i] for i in t.token_indices),
                    top1_logprobs=tuple(top1_logprobs[i] for i in t.token_indices),
                    top2_logprobs=tuple(top2_logprobs[i] for i in t.token_indices),
                    legal_actions=t.legal_actions,
                    action_valid=t.action_valid,
                    # anchor 始终指动作执行前的状态；即使该动作直接解题，也不能把此状态标成终止状态。
                    terminal=False,
                )
                for t in self.turns
            ),
        )


async def run_episode(
    io,
    env,
    *,
    task_id: str,
    seed: int,
    trajectory_id: str,
    max_turns: int = 8,
    response_budget: int = 2048,
    action_token_limit: int = 32,
    scripted_actions: Optional[Mapping[int, GeneratedAction]] = None,
) -> Episode:
    """通过注入的异步模型/tokenizer 接口运行真实环境。

    ``io`` provides initial(messages), generate(stream, limit),
    assistant(stream, generated), and context(stream, previous_messages, updated_messages).
    No environment feedback is appended after the final assistant action, so terminal
    reward attaches to an assistant token rather than an observation token.
    """
    if min(max_turns, response_budget, action_token_limit) <= 0:
        raise ValueError("Episode limits must be positive")
    state = env.reset(task_id, seed)
    messages = initial_messages(state)
    stream = await io.initial(messages)
    stream.validate()
    turns, prefix = [], []
    reason = "horizon"
    for _ in range(max_turns):
        if state.terminal:
            reason = "terminal"
            break
        # 为模板边界 token 预留 response 空间；不能静默截断已记录的动作 token。
        room = response_budget - len(stream.response_mask) - 16
        if room <= 0:
            reason = "token_budget"
            break
        turn_id = len(turns)
        limit = min(action_token_limit, room)
        if scripted_actions is not None and turn_id in scripted_actions:
            generated = scripted_actions[turn_id]
            if len(generated.token_ids) > limit:
                raise ValueError("Scripted action exceeds the generation limit")
        else:
            generated = await io.generate(stream, limit)
        if not generated.token_ids:
            reason = "empty_generation"
            break
        merged = await io.assistant(stream, generated)
        indices = validate_merge(stream, merged, assistant=True)
        if not indices or len(merged.response_mask) > response_budget:
            raise ValueError("Assistant merge exceeded budget or produced no trained tokens")
        # 格式错误的模型输出仍占用一个 turn，并作为非法动作处理；这里不猜测用户想走哪一步。
        text = generated.text.strip().lower()
        action = text if text in ACTIONS else "invalid"
        next_state = env.step(action)
        turns.append(
            EpisodeTurn(
                action=action,
                raw_text=generated.text,
                action_prefix=tuple(prefix),
                observation=state.observation,
                state_hash=state.state_hash,
                next_observation=next_state.observation,
                next_state_hash=next_state.state_hash,
                legal_actions=state.legal_actions,
                action_valid=action in state.legal_actions,
                token_indices=indices,
                reward=next_state.reward,
                done_after=next_state.terminal,
                generated_token_ids=generated.token_ids,
                generated_logprobs=generated.logprobs,
            )
        )
        prefix.append(action)
        state, stream = next_state, merged
        messages.append({"role": "assistant", "content": generated.text})
        if state.terminal:
            reason = "terminal"
            break
        if len(turns) == max_turns:
            break
        updated = messages + [{"role": "user", "content": observation_message(state)}]
        context = await io.context(stream, messages, updated)
        validate_merge(stream, context, assistant=False)
        # 避免 response 以可能被误判为终止奖励来源的 observation 收尾。
        if len(context.response_mask) + 16 >= response_budget:
            reason = "token_budget"
            break
        messages, stream = updated, context
    if not turns:
        raise ValueError(f"No assistant action generated: {reason}")
    # 反馈后若模型没有生成任何 token，则丢弃尾部未训练上下文，避免制造空动作轮。
    last = turns[-1].token_indices[-1] + 1
    stream = TokenStream(
        stream.token_ids[: stream.prompt_length + last],
        stream.response_mask[:last],
        stream.logprobs[:last] if stream.logprobs is not None else None,
    )
    return Episode(
        trajectory_id, task_id, seed, max_turns, stream, tuple(turns), state.reward, reason
    )
