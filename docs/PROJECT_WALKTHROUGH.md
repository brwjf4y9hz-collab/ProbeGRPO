# ProbeGRPO 项目导览

本项目验证多轮环境中的反事实探测如何进入 GRPO 更新链路，并同时记录训练结果与额外采样成本。

## 问题与方法

终局奖励判断整条轨迹是否完成任务，却难以区分各轮动作的局部影响。ProbeGRPO 固定决策点此前的交互前缀，比较原动作与替代动作的后续结果，将奖励差作为局部 credit。

一次训练批次包含六个步骤：

1. 生成主 rollout，提取动作、观察、状态和 assistant token 索引。
2. 在预算内选取 anchor，重置环境并重放此前动作。
3. 核验两个分支状态与观察一致，执行事实动作与合法替代动作。
4. 成对后续采样得到奖励差；状态不一致或动作无效时跳过。
5. 将有效差值归一化并加入对应 turn 的 advantage，保留其他 token 的原始值。
6. actor 更新，记录额外开销，在共同留出关卡上评估。

[数据契约](ARCHITECTURE.md) · [训练接入手册](SOKOBAN_AGENTLOOP.md)

## 实现选择

| 选择 | 目的 | 边界 |
|---|---|---|
| 确定性前缀回放与状态校验 | 保证分支从同一状态开始 | 依赖可重置、可验证环境 |
| 局部 assistant token mask | 限制 credit 的作用范围 | 需要与 token 序列严格对齐 |
| 预算与调度器 | 比较选点策略及开销 | 当前不是全局 top-B |
| 核心包与 verl 解耦 | 先在 CPU 验证，再接训练框架 | GPU 训练需独立环境 |

## 结果解释

公开 Sokoban 实验使用 Qwen3.5-2B，512 个训练关卡、128 个共同留出关卡，四方法各三个种子、50 次更新。

LinearUCB-B2 历史平均成功率 30.99%，GRPO 为 23.96%，配对差值约 7.03 个百分点；额外 rollout token 为自身主 rollout token 的 51.64%。Surprisal 均值更高，Random 的单位额外 token 诊断更高，各指标需要分别解释。

seed17 GRPO 配置差异、三种子规模、非等总预算与最终权重缺失，限制结论范围。现有记录支持描述性比较，不能据此宣称统计显著性或最优选点器。

[结果图与逐种子数据](results/README.md) · [实验协议](../experiments/results/public_sokoban_main_v1/README.md)

## 复核顺序

- 用 [CPU smoke](../scripts/run_cpu_smoke.sh) 检查回放、探测与局部更新。
- 查看 [advantage.py](../src/probegrpo/advantage.py) 中的零差值、有效性和 mask 处理。
- 查看 [verl sidecar](../src/probegrpo/integration/verl_v1_sidecar.py) 的训练批次对齐。
- 从[保存清单](../repro/ARTIFACTS.md)定位记录，区分早期 smoke 权重与正式数据。
- 从[问题复盘](incidents/README.md)核对故障、处理依据与剩余限制。

CPU 演示、GPU 链路验证与正式训练是不同验证层次。后续可以补严格同配置、等总预算比较，并检查新任务效果。
