# ProbeGRPO 学习复现路线

这份路线用于边运行、边读代码、边核对证据。仓库已有完整的三种子公开 Sokoban 结果；学习时从最小可运行链路开始，不能把本地 CPU 演示当作重新复现了模型训练结果。

## 检查点 1：看见一条多轮轨迹（当前）

在项目根目录运行：

```bash
make agent-smoke
```

观察输出中的两个 `turn`：动作依次为 `up`、`up`，两次 `replay_match=True`，最终 `reward=1.0`。第一步的 assistant token 位置是 `(0, 1)`，第二步是 `(140, 141)`；中间的大段位置属于环境 observation，不能获得 assistant 动作的局部 credit。

沿这条线读代码：`src/probegrpo/agent_smoke.py` → `src/probegrpo/agent_episode.py` → `src/probegrpo/envs/sokoban.py`。验收是能够用自己的话解释：动作前缀为什么只包含当前动作之前的动作，第二个 turn 的 token 为什么没有紧贴第一个 turn。

本机验证记录（2026-09-24）：`make check` 运行 78 个单元测试，通过 76 个、跳过 2 个需要 PyTorch 的测试；`probegrpo.smoke` 和 `probegrpo.agent_smoke` 均通过。此记录只证明 CPU 逻辑与脚本演示可运行，用户理解情况尚待检查。

## 检查点 2：看见局部 advantage

运行 `make smoke`，比较输出中的 `base_advantages`、`final_advantages` 和 `probe_turn_masks`。接着读 `src/probegrpo/advantage.py` 中的 `blend_probe_advantages`，以及 `tests/test_advantage.py` 的 `test_only_selected_turn_tokens_change`。验收是能指出哪些 token 改变、改了多少，以及为什么 `delta=0` 时所有 token 都不变。

## 检查点 3：真实训练链路

在满足 `docs/GPU_RUNBOOK.md` 要求的 Linux NVIDIA GPU 上，先按 runbook 检查并建立固定版本的 verl 运行环境，再运行 `scripts/run_verl_grpo_smoke.sh`。用日志和 checkpoint 分别核对 rollout、reward、GRPO advantage、actor loss、梯度和参数更新。该脚本默认关闭定期验证；若要观察完整的 `dataset → rollout → reward → advantage → loss → backward → optimizer.step → validation`，运行时显式设置 `trainer.test_freq=1` 并使用验证数据。无 GPU 时停留在 CPU 检查点，不声称完成了训练复现。

## 检查点 4：ProbeGRPO 和正式结果

按 `docs/SOKOBAN_AGENTLOOP.md` 检查无 probe 的 GRPO、真实 probe、turn-local credit 和公开数据对比。核对 `experiments/results/public_sokoban_main_v1/` 中的配置、种子、汇总与成本。验收是能解释同一预算下 GRPO 与 ProbeGRPO 的差异，并区分已有记录与自己实际重跑得到的结果。
