#!/usr/bin/env python3
"""Export sign-balanced, deterministic v1 examples with replay and assistant-mask checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probegrpo.envs.sokoban import SokobanEnv


def export(run: Path, output: Path):
    manifest = json.loads((run / "run_manifest.json").read_text())
    cases = {}
    paths = sorted((run / "rollouts/episodes").glob("step-*/*.json"),
                   key=lambda p: (int(p.parent.name.removeprefix("step-")), p.name))
    for path in paths:
        raw = path.read_bytes()
        episode = json.loads(raw)
        probe = episode.get("debug_probe") or {}
        if (episode.get("dataset_split") != "train" or probe.get("skipped_reason")
                or not probe.get("state_match") or "anchor_turn_id" not in probe):
            continue
        delta = float(probe["delta"])
        sign = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
        if sign in cases:
            continue
        index = int(probe["anchor_turn_id"])
        turn = episode["turns"][index]
        env = SokobanEnv(max_steps=int(episode["max_horizon"]))
        before = env.replay(episode["task_id"], int(episode["seed"]), turn["action_prefix"])
        if before.state_hash != turn["state_hash"] or before.observation != turn["observation"]:
            raise ValueError("source case failed deterministic replay")
        after = env.step(probe["counterfactual_action"])
        factual = SokobanEnv(max_steps=int(episode["max_horizon"])).replay(
            episode["task_id"], int(episode["seed"]), [*turn["action_prefix"], turn["action"]])
        if factual.state_hash != turn["next_state_hash"]:
            raise ValueError("source factual transition mismatch")
        indices = turn["token_indices"]
        mask = episode["stream"]["response_mask"]
        if not indices or not all(mask[i] == 1 for i in indices):
            raise ValueError("anchor is not confined to assistant tokens")
        cases[sign] = {
            "sign": sign, "training_seed": manifest["seed"],
            "source": str(path.relative_to(run)), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "task_id": episode["task_id"], "environment_seed": episode["seed"],
            "max_horizon": episode["max_horizon"], "anchor_turn_id": index,
            "action_prefix": turn["action_prefix"], "board_before_anchor": before.observation,
            "factual_action": probe["factual_action"], "counterfactual_action": probe["counterfactual_action"],
            "factual_next_board": factual.observation, "counterfactual_next_board": after.observation,
            "factual_reward": probe["factual_reward"], "counterfactual_reward": probe["counterfactual_reward"],
            "delta": delta, "probe_tokens": probe["additional_rollout_tokens"],
            "assistant_response_indices": indices, "scheduler": probe["scheduler"],
            "main_actions": [t["action"] for t in episode["turns"]],
        }
        if len(cases) == 3:
            break
    if len(cases) != 3:
        raise ValueError("cannot find all positive/negative/zero cases; do not fabricate a missing case")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"selection": "first valid positive/negative/zero by numeric step then filename; illustrative, not representative",
               "run": str(run), "model_revision": manifest["model_revision"],
               "scope": "historical v1 model-generated reward records; single-action boards independently replayed on CPU; no suffix regenerated",
               "cases": [cases[sign] for sign in ("positive", "negative", "zero")]}
    (output / "cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 真实轨迹案例：正、负、零 credit", "",
             "来源为 v1 真实模型实验；按 step 和文件名确定性选取各符号的第一个有效案例，不代表频率或性能。",
             "棋盘只做确定性重放和单动作转移；两条完整 probe suffix 未保存，不能展示或声称重新验证其整条轨迹。",
             "保存的 probe 回报与原主轨迹最终回报不一定相同，因为 factual suffix 也重新采样。", ""]
    for case in payload["cases"]:
        lines += [f"## {case['sign']}：ΔReward={case['delta']:+g}", "",
                  f"来源：`{case['source']}`；SHA-256：`{case['source_sha256']}`。", "",
                  f"scheduler=`{case['scheduler']}`，anchor turn（从 0 起）={case['anchor_turn_id']}；"
                  f"动作 `{case['factual_action']}` → `{case['counterfactual_action']}`。", "",
                  "anchor 前棋盘（已校验状态哈希与观察一致）：", "", "```text", case["board_before_anchor"], "```", "",
                  f"原动作后棋盘：\n\n```text\n{case['factual_next_board']}\n```", "",
                  f"替代动作后棋盘：\n\n```text\n{case['counterfactual_next_board']}\n```", "",
                  f"保存的 factual / counterfactual suffix 回报：{case['factual_reward']} / "
                  f"{case['counterfactual_reward']}；额外生成 {case['probe_tokens']} tokens。", "",
                  f"仅 assistant response 索引 `{case['assistant_response_indices']}` 可接收 credit，"
                  "环境 observation 不可接收。", ""]
    lines += ["λ=0.5 且二值 Δ∈{-1,0,1} 时，这三种情况的理论局部 advantage 增量分别为 +0.5、-0.5、0。",
              "这是按公式计算的说明，不是从原始 actor 梯度中恢复的数值；完整 GRPO advantage/实际 loss 不在本案例中伪造。", "",
              "复核命令：`PYTHONPATH=src python scripts/export_trajectory_demo.py <原始 run 目录> --output-dir <新目录>`。", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Exported and replay-checked {len(cases)} historical cases to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
