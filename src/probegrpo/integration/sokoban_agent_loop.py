"""Pinned-verl AgentLoop adapter; importing this module requires the GPU runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from probegrpo.agent_episode import GeneratedAction, TokenStream, run_episode
from probegrpo.envs.sokoban import SokobanEnv
from probegrpo.episode_probing import probe_episode
from probegrpo.probe_mode import resolve_probe_mode


class VerlTokenIO:
    def __init__(self, agent, sampling_params, request_id, priority):
        self.agent = agent
        self.params = dict(sampling_params)
        self.request_id, self.priority = request_id, int(priority)
        self.extra_fields = {}
        self.generate_seconds = 0.0

    async def initial(self, messages):
        ids = await self.agent.ct_build_initial_tokens(messages)
        return TokenStream(tuple(ids))

    async def generate(self, stream, limit):
        params = {**self.params, "max_tokens": limit}
        start = time.monotonic()
        output = await self.agent.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=list(stream.token_ids),
            sampling_params=params,
            priority=self.priority,
        )
        self.generate_seconds += time.monotonic() - start
        for key, value in (output.extra_fields or {}).items():
            if key == "min_global_steps" and key in self.extra_fields:
                value = min(self.extra_fields[key], value)
            if key == "max_global_steps" and key in self.extra_fields:
                value = max(self.extra_fields[key], value)
            self.extra_fields[key] = value
        return GeneratedAction(
            tuple(output.token_ids),
            self.agent.tokenizer.decode(output.token_ids, skip_special_tokens=True),
            tuple(output.log_probs) if output.log_probs is not None else None,
        )

    async def assistant(self, stream, generated):
        previous_probs = list(stream.logprobs) if stream.logprobs is not None else None
        if previous_probs is None and generated.logprobs is not None:
            if any(stream.response_mask):
                raise ValueError("Rollout logprobs appeared after an unscored assistant turn")
            previous_probs = [0.0] * len(stream.response_mask)
        if previous_probs is not None and generated.logprobs is None:
            raise ValueError("Missing rollout logprobs for an assistant turn")
        result, mask, probs = await self.agent.ct_merge_assistant_token(
            list(stream.token_ids),
            list(generated.token_ids),
            list(stream.response_mask),
            previous_probs,
            assistant_logprobs=list(generated.logprobs) if generated.logprobs is not None else None,
        )
        return TokenStream(
            tuple(result.token_ids), tuple(mask), tuple(probs) if probs is not None else None
        )

    async def context(self, stream, previous, updated):
        result, mask, probs = await self.agent.ct_merge_context_msg(
            previous,
            updated,
            list(stream.token_ids),
            list(stream.response_mask),
            list(stream.logprobs) if stream.logprobs is not None else None,
        )
        return TokenStream(
            tuple(result.token_ids), tuple(mask), tuple(probs) if probs is not None else None
        )


class SokobanAgentLoop(AgentLoopBase):
    """Uses real model generations and environment reward, without a language-model judge."""

    async def run(self, sampling_params, priority=0, **kwargs):
        info = kwargs.get("extra_info", {})
        task_id, seed = str(info["task_id"]), int(info["env_seed"])
        trajectory_id = f"{kwargs['uid']}_{kwargs.get('session_id', 0)}"
        max_turns = int(info.get("max_turns", 8))
        mode = resolve_probe_mode(
            self.config.get("probe", {}),
            session_id=int(kwargs.get("session_id", 0)),
            debug_probe=os.environ.get("PROBEGRPO_DEBUG_PROBE") == "1",
        )
        io = VerlTokenIO(self, sampling_params, trajectory_id, priority)
        episode = await run_episode(
            io,
            SokobanEnv(max_steps=max_turns),
            task_id=task_id,
            seed=seed,
            trajectory_id=trajectory_id,
            max_turns=max_turns,
            response_budget=int(self.rollout_config.response_length),
        )
        payload = episode.as_dict()
        # A narrow training gate: at most one random anchor in session zero.
        # The trainer decides whether to apply its credit after standard GRPO.
        if mode.run_probe:
            probe = await self._random_probe(episode, sampling_params, priority, trajectory_id)
            probe["scheduler"] = mode.scheduler
            probe["credit_requested"] = mode.credit_requested
            probe["advantage_applied"] = False  # The actor update has not happened yet.
            payload["debug_probe"] = probe
        # Standard verl JSONL omits custom extra_fields. Keep a separate per-episode sidecar.
        step = int(kwargs.get("global_steps", 0))
        root = Path(self.config.trainer.rollout_data_dir) / "episodes" / f"step-{step}"
        root.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(trajectory_id.encode()).hexdigest() + ".json"
        with (root / filename).open("x") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        stream = episode.stream
        response_length = len(stream.response_mask)
        return AgentLoopOutput(
            prompt_ids=list(stream.token_ids[: stream.prompt_length]),
            response_ids=list(stream.token_ids[stream.prompt_length :]),
            response_mask=list(stream.response_mask),
            response_logprobs=list(stream.logprobs) if stream.logprobs is not None else None,
            reward_score=episode.final_reward,
            num_turns=2 * len(episode.turns),
            metrics={"generate_sequences": io.generate_seconds},
            extra_fields={
                **io.extra_fields,
                "probegrpo_episode_json": json.dumps(payload),
                "assistant_turns": len(episode.turns),
                "response_token_count": response_length,
            },
        )

    async def _random_probe(self, episode, sampling_params, priority, trajectory_id):
        seed = int.from_bytes(hashlib.sha256(trajectory_id.encode()).digest()[:4], "big")
        rng = random.Random(seed)
        candidates = [
            index
            for index, turn in enumerate(episode.turns)
            if turn.action in turn.legal_actions and len(set(turn.legal_actions)) >= 2
        ]
        if not candidates:
            return {"skipped_reason": "no_eligible_anchor", "delta": 0.0}
        anchor_turn_id = rng.choice(candidates)
        turn = episode.turns[anchor_turn_id]
        alternatives = sorted(set(turn.legal_actions) - {turn.action})
        alternative_action = rng.choice(alternatives)
        token_ids = tuple(self.tokenizer.encode(alternative_action, add_special_tokens=False))
        if self.tokenizer.eos_token_id is not None:
            token_ids += (int(self.tokenizer.eos_token_id),)
        alternative = GeneratedAction(
            token_ids, alternative_action, (0.0,) * len(token_ids)
        )

        def io_factory(sampling_seed, branch_name):
            return VerlTokenIO(
                self,
                {**sampling_params, "seed": sampling_seed},
                f"{trajectory_id}:probe:{branch_name}",
                priority,
            )

        result = await probe_episode(
            episode,
            anchor_turn_id,
            alternative,
            env_factory=lambda: SokobanEnv(max_steps=episode.max_horizon),
            io_factory=io_factory,
            sampling_seed=seed,
            response_budget=int(self.rollout_config.response_length),
            timeout_seconds=float(os.environ.get("PROBEGRPO_PROBE_TIMEOUT_S", "180")),
        )
        return {
            **asdict(result),
            "anchor_turn_id": anchor_turn_id,
            "sampling_seed": seed,
        }
