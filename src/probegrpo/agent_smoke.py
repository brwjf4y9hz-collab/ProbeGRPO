"""本模块提供无需模型权重的脚本化 episode 演示。固定动作和人工构造的统计量只用于说明数据流，不代表真实模型采样或不确定性。"""

import asyncio

from .agent_episode import GeneratedAction, TokenStream, run_episode
from .envs.sokoban import SokobanEnv
from .rollout import extract_turn_records


class ScriptedTokenIO:
    """仅供 CPU 验证的字符级 tokenization 和固定动作；这不是 LLM rollout。"""

    def __init__(self, actions):
        self.actions = iter(actions)

    async def initial(self, messages):
        return TokenStream(tuple(map(ord, str(messages))))

    async def generate(self, stream, limit):
        text = next(self.actions, "invalid")[:limit]
        return GeneratedAction(tuple(map(ord, text)), text, (-0.1,) * len(text))

    async def assistant(self, stream, generated):
        return TokenStream(
            stream.token_ids + generated.token_ids,
            stream.response_mask + (1,) * len(generated.token_ids),
            (stream.logprobs or ()) + (generated.logprobs or ()),
        )

    async def context(self, stream, previous, updated):
        tokens = tuple(map(ord, updated[-1]["content"]))
        return TokenStream(
            stream.token_ids + tokens,
            stream.response_mask + (0,) * len(tokens),
            (stream.logprobs or ()) + (0.0,) * len(tokens),
        )


async def demo():
    episode = await run_episode(
        ScriptedTokenIO(["up", "up"]),
        SokobanEnv(),
        task_id="push-up",
        seed=17,
        trajectory_id="cpu-demo",
    )
    stream = episode.stream
    print("SCRIPTED CPU demo: no model generation or measured entropy")
    print(
        f"T={len(stream.response_mask)} response tokens; "
        f"assistant={sum(stream.response_mask)}; "
        f"observation={stream.response_mask.count(0)}"
    )
    for i, turn in enumerate(episode.turns):
        replayed = SokobanEnv().replay(episode.task_id, episode.seed, turn.action_prefix)
        assert replayed.state_hash == turn.state_hash
        print(
            f"turn={i} prefix={turn.action_prefix} action={turn.action} "
            f"assistant_indices={turn.token_indices} replay_match=True"
        )
        print(turn.next_observation)
    # 显式传入的合成 actor 统计仅演示数据桥接方式，不代表模型真实不确定性。
    count = len(stream.response_mask)
    trace = episode.to_trace(
        entropies=[0.5] * count, top1_logprobs=[-0.1] * count, top2_logprobs=[-0.6] * count
    )
    records = extract_turn_records(trace)
    print(f"TrajectoryTrace -> {len(records)} TurnRecords (synthetic statistics)")
    print(f"reward={episode.final_reward} stop={episode.stop_reason}")
    assert episode.final_reward == 1.0
    print("ProbeGRPO Sokoban episode CPU smoke passed")


def main():
    asyncio.run(demo())


if __name__ == "__main__":
    main()
