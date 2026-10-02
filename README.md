# ProbeGRPO

ProbeGRPO 是一个面向 Agent-RL 实习作品集的项目，展示从多轮环境交互到细粒度策略更新的完整流程。项目在 GRPO 上增加了**带预算的反事实 turn 探测**：重放选中的决策状态，尝试另一个合法动作，测量环境奖励变化，再将所得局部信号用于对应的 assistant turn。

项目目标是提供一个可信、可检查的工程实现，不声称反事实 credit assignment 本身是新方法。项目特色是可复用的小型核心、确定性重放校验、预算感知的选点策略、当前版本 verl 接入、源自 RAGEN 的任务适配器，以及明确的成本记录。

## 保存与复现入口（2026-10-02）

代码、配置和小型结果保存在 Git；框架使用固定 commit 和最小 patch；模型、数据及原始输出保存在独立归档，并记录 SHA-256。入口见[保存清单](repro/ARTIFACTS.md)、[干净环境复现说明](REPRODUCE.md)、[仓库盘点](REPO_AUDIT.md)和[核验记录](repro/validation/README.md)。

已重新核对 12 次历史运行中的 1,536 条最终评估记录。主实验当时没有保存最终模型权重；现存的大型 checkpoint 属于早期 GSM8K smoke run。整理期间尚未重新进行 GPU 训练。

## 已实现内容

- 与框架无关的 `ReplayableEnv` 接口，支持确定性前缀重放和状态哈希。
- 结构化的 `TurnRecord`、`Anchor`、`ProbeResult` 和实验指标类型。
- 随机、chosen-token surprisal 和在线 LinearUCB anchor 选点器。
- 带严格重放校验的 factual/counterfactual 成对后缀探测。
- 支持 Probe credit 的 GRPO advantage shaping，并精确标记 assistant turn token。
- 延迟导入 torch/verl 的适配器，核心包不依赖训练框架。
- 确定性 Sokoban 环境、公开 RAGEN 数据适配器和一条命令即可运行的 CPU smoke test。
- 固定版本的当前 verl 初始化脚本，以及面向单张 48 GB GPU、使用 `Qwen/Qwen3.5-2B` 的五步 GRPO gate。

维护者已运行五个标准 GRPO update、从 step 5 恢复到 step 6，并核验一个修正后的单步运行：8 条轨迹且梯度非零。不同配置、故障及证据边界见 [GPU gate 记录](experiments/environment/2026-09-19-gpu-gate.md)。这不是 ProbeGRPO 性能结果。

当前 verl Sokoban AgentLoop 已完成真实 Qwen rollout 和带/不带 probe credit 的 actor update。三个手工关卡仅用于 smoke test。正式实验使用固定版本的公开 RAGEN 数据，去重棋盘布局、排除训练/测试重叠，并为每个选中棋盘记录精确最短路径 oracle。操作步骤见 [AgentLoop 运行手册](docs/SOKOBAN_AGENTLOOP.md)。

## 主要结果

公开 Sokoban 对比实验中，每种方法使用 `Qwen/Qwen3.5-2B` 训练 50 个 update，运行三个随机种子。下表是在相同 128 个 held-out 棋盘上的 seed 平均值 ± seed 间样本标准差，种子为 17、42、101。

| 方法 | 最终成功率 | 相对 GRPO 的配对提升 | 额外 rollout token |
|---|---:|---:|---:|
| GRPO | 24.0% ± 2.0% | - | 0.0% |
| Random-B2 | 29.2% ± 4.6% | +5.2 ± 6.5 个百分点 | 51.2% |
| Surprisal-B2 | 31.2% ± 3.4% | +7.3 ± 4.3 个百分点 | 57.3% |
| **LinearUCB-B2** | **31.0% ± 1.2%** | **+7.0 ± 0.8 个百分点** | **51.6%** |

![公开 Sokoban 三种子实验结果](docs/assets/public_sokoban_main_v1.svg)

LinearUCB 在三个种子中都高于对应 GRPO，且 probe 方法中的跨种子波动最低。Surprisal 的均值高 0.2 个百分点，但相对 seed 间波动很小，成本也更高，不能视为有意义的胜出。每 1,000 个 probe token 找到的高影响 anchor 数量上，Random 更高。因此，这些结果支持“turn-local counterfactual credit 有正向信号”和“本次 LinearUCB 结果较稳定”，**不支持**“LinearUCB 是最高效的 anchor 选点器”。完整结果、协议偏差和证据边界见[结果记录](experiments/results/public_sokoban_main_v1/README.md)。

## 系统流程

```text
主 rollout：每题 K=4 条轨迹
       |
       v
提取 TurnRecord -----> anchor 选点器
                            | 预算 B=2
                            v
                       确定性重放
                         /      \
                factual 后缀    counterfactual 后缀
                         \      /
                       奖励差 delta
                            |
                            v
       trajectory GRPO advantage + turn-local probe credit
                            |
                            v
                        策略更新
```

数据契约和接入边界见 [系统架构](docs/ARCHITECTURE.md)。

## 快速开始（CPU，无需下载模型）

核心包和 smoke test 只依赖 Python 标准库。

```bash
cd ProbeGRPO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
make check
```

干净 clone、固定的 `verl` patch、数据/模型来源及 GPU smoke/run 要求见 [REPRODUCE.md](REPRODUCE.md)。

如果已有环境，只需激活环境并运行 `make check`。本机 macOS 环境用于开发和 CPU 测试；verl、PyTorch CUDA 和 vLLM 应在独立的 NVIDIA Linux 环境中安装。

## CPU smoke 验证

不安装项目也可以单独检查 CPU 数据流：

```bash
bash scripts/run_cpu_smoke.sh
```

成功时末行应为：

```text
ProbeGRPO CPU data-flow smoke test passed
```

## 当前 verl + Qwen3.5

默认模型为 [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)。RAGEN 发布版本的训练依赖较旧，不能直接满足该模型，因此 ProbeGRPO 固定了一个较新的 verl revision 及其依赖锁，并显式覆盖 NumPy 版本为 2.3.5。在 AutoDL 上的一台 48 GB NVIDIA GPU Linux 实例中，可按以下步骤准备环境：

```bash
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
export MODEL_PATH=/path/to/persistent-workspace/models/Qwen3.5-2B-15852e8
/path/to/persistent-workspace/probegrpo-runtime/verl/.venv/bin/python scripts/download_model.py --output-dir "$MODEL_PATH"
bash scripts/check_verl_stack.sh /path/to/persistent-workspace/probegrpo-runtime/verl
bash scripts/run_verl_grpo_smoke.sh /path/to/persistent-workspace/probegrpo-runtime/verl
```

首次付费运行刻意选 GSM8K，而不是 Agent 环境。先验证模型加载、vLLM 生成、LoRA/FSDP2 更新、grouped GRPO、checkpoint 保存和恢复训练，再引入 Sokoban 的复杂性。详见 [GPU 运行手册](docs/GPU_RUNBOOK.md)。

ProbeGRPO 的适配器要求 trainer batch 含有以下张量：

- `advantages`：`[batch, tokens]`
- `probe_turn_masks`：`[batch, anchors, tokens]`
- `probe_deltas`：`[batch, anchors]`
- `probe_valid`：`[batch, anchors]`

实现会将有效 delta 除以本 batch 内 `max(1, 最大绝对 delta)`，并把 `lambda * normalized_delta * turn_mask` 加到原始 advantage 上。事实和反事实奖励相同（delta 为零）时，credit 也始终为零。

## 项目里程碑

1. **核心正确性：**`make check` 通过，测试确定性重放及 token mask 下的 advantage。
2. **GPU 栈 gate：**Qwen3.5-2B 在当前 verl 上完成五次标准 GRPO update，并能恢复训练。
3. **Agent rollout：**RAGEN 衍生的 Sokoban 任务完成当前 verl 多轮 `AgentLoop` rollout 和确定性前缀重放。
4. **同预算比较：**GRPO、Random、Surprisal、LinearUCB 均完成三个种子。
5. **作品集整理：**结果表和成本图已完成；轨迹演示及可选的 WebShop 扩展尚未完成。

作为实习作品集，解释清楚的负结果也有价值。不要捏造提升；如果额外 probe 改善了 credit 诊断，却没有提升最终奖励，也应如实报告。

在一张 GPU 上运行可复现的消融：

```bash
bash scripts/run_sokoban_ablation.sh /root/autodl-tmp/probegrpo-runtime/verl main
```

将 `SEED` 设为 17、42 或 101。运行器会生成每个 seed 的 JSON 和 Markdown 表格。执行 `make results` 可重新生成已提交的三种子汇总和 SVG。当前 budget 实现会在每个 prompt group 的四条 episode 中先选前 `B` 条，再在每条选中的 episode 内选一个 turn；它还不是对整组 episode/turn 候选做全局 top-B 排序。

## 评估指标

- 任务奖励和成功率；
- 非法动作率和平均 episode 长度；
- reward/advantage 方差；
- rollout token 总数、GPU 小时数和墙钟时间；
- 每 1,000 个额外 rollout token 找到的高影响 anchor 数；
- Sokoban 探测排序与穷举 counterfactual oracle 的相关性。

## 仓库规范

- 原始轨迹、checkpoint 和 W&B 文件放在 Git 忽略的 `artifacts/`、`outputs/` 或 `wandb/` 中。
- 完整可恢复 checkpoint 保存在 Git 以外。只有完成真实导出与核验后才发布便携 adapter；目前没有主实验 adapter。
- 不提交 API key、cookie、WebShop session 或未脱敏外部轨迹。
- 结果必须包含精确配置、Git revision、种子和失败运行。
- 环境准备时从固定 commit 获取上游 `verl`；本仓库不 vendoring 整个框架。ProbeGRPO 对 trainer 的改动保存在 `patches/verl-probegrpo.patch`，由 `scripts/bootstrap_verl.sh` 应用。

## 简历描述

> 在当前 verl 上构建了 ProbeGRPO Qwen3.5 Agent-RL 系统，支持确定性 counterfactual 重放、预算感知的 anchor 选点和 turn-local advantage shaping；公开 Sokoban 三种子结果从 24.0% ± 2.0% 提升至 31.0% ± 1.2%，额外 rollout token 成本为 1.516 倍。
