# ProbeGRPO 技术报告

> 状态：已完成公开 Sokoban 三种子作品集实验；WebShop 属于可选扩展。

## 摘要

长程语言模型 Agent 通常只会得到整条轨迹的终局奖励。GRPO 能比较同一 prompt 下采样出的多条轨迹，但同一轨迹中每个生成的动作 token 仍共享轨迹级信号。ProbeGRPO 增加有限的环境交互预算：选取少数 assistant turn，确定性重放其动作前缀，替换一个合法动作，再把事实分支与反事实分支的终局奖励差只用于该 turn。系统在当前 verl 上实现，使用 Qwen3.5-2B LoRA actor 和可重放的 Sokoban AgentLoop。

在固定校验和的公开 RAGEN Sokoban 划分上，四种方法各训练 50 个 update、使用三个随机种子。LinearUCB-B2 的 held-out 成功率从 GRPO 的 `24.0% ± 2.0%` 提高至 `31.0% ± 1.2%`，配对提升 `7.0 ± 0.8` 个百分点，额外 rollout token 为 `51.6%`。Surprisal-B2 的均值相近但成本更高；Random-B2 的波动更大。这些结果支持一个可复现的工程结果，不构成统计显著性结论，也不声称反事实 credit assignment 本身是新颖的。

## 1. 问题与设计目标

对包含 assistant turns `a_1 ... a_T`、最终奖励为 `R` 的 episode，trajectory-level GRPO 会把同一个相对轨迹 advantage 分配给所有 response token。这样做成本低，但信号较粗：失败轨迹中的正确早期动作会被后续错误一起惩罚，成功轨迹中的无关动作却会一起得到正向信号。

ProbeGRPO 聚焦一个较窄的工程问题：在固定额外 rollout 预算下，真实环境重放能否在不使用 LLM judge 的情况下提供有用的 turn-local 信号？设计要求如下：

1. 事实与反事实后缀必须从同一个经过验证的状态开始；
2. 明确记录额外 rollout token 成本；
3. probe credit 只修改选中的 assistant turn；
4. budget 为零或 credit 系数为零时，精确回退到普通 GRPO；
5. 调度器比较使用相同的主 rollout、数据集、模型和训练预算。

## 2. 系统

### 2.1 主 AgentLoop

每个 update 对四个 prompt 各采样四条轨迹。Qwen actor 在每个 assistant turn 生成一个动作字符串；Sokoban 将新棋盘作为环境 observation 返回。每条 turn 记录当前动作之前的 action prefix、state hash、合法动作、所采动作 token 的 log-probability、response token 索引、动作是否合法及最终轨迹奖励。环境 observation 也占 response 位置，但其 assistant mask 为零。

### 2.2 反事实 probe

对 anchor turn `t`，创建两个新环境，使用同一 task ID 和 seed 重置，再重放 `a_1 ... a_(t-1)`。如果重建的 observation 或 state hash 与记录不一致，就丢弃该 probe。事实分支执行 `a_t`；反事实分支执行另一个合法动作。随后两个分支都用相同的采样 seed 生成后缀。局部差值为：

```text
delta_t = factual_terminal_reward - counterfactual_terminal_reward
```

正值表示原始动作在本次被测条件下优于所选替代动作；负值表示替代动作更好。零差值属于有效观测，但不会产生更新增量。

### 2.3 Advantage 接入

Sidecar 将稀疏 probe 记录转换为稠密张量：

```text
probe_turn_masks [N, A, T]
probe_deltas     [N, A]
probe_valid      [N, A]
```

verl 算完标准 GRPO advantage 后、actor update 前，ProbeGRPO 应用：

```text
A_final = A_GRPO + lambda * normalize(delta) * turn_mask
```

其中 `lambda=0.5`。Mask 只覆盖选中 assistant action token，不包含 prompt、环境 observation、padding 或其他 turn。现有集成测试覆盖 budget 为零、lambda 为零、无有效 probe 和 delta 全为零时的回退行为。

### 2.4 调度器

- **Random-B2：**按轨迹位置分层随机选择 turn。
- **Surprisal-B2：**优先选择 chosen-token surprisal 高的动作。此处不称作 entropy，因为 rollout 后端没有提供完整 next-token 分布。
- **LinearUCB-B2：**在线线性 contextual bandit，根据额外 rollout token 归一化后的绝对 probe delta 估计价值，并加入不确定性 bonus 和随机 warm-up。

当前 budget=2 的实现会在每个 prompt group 的四条轨迹中先选择前两条，再在每条选中轨迹内选一个 turn。整组 episode/turn 候选的全局 top-B 选择尚未实现。

## 3. 实验配置

| 组件 | 设置 |
|---|---|
| 策略模型 | `Qwen/Qwen3.5-2B` |
| 参数高效微调 | BF16 LoRA，rank 32，alpha 64 |
| Trainer | 当前 verl，FSDP2 actor 与 vLLM rollout |
| 硬件 | 单张 RTX 4090 48 GB |
| 数据集 | 固定版本的公开 RAGEN Sokoban 数据 |
| 划分 | 512 个训练棋盘 / 128 个 held-out 测试棋盘 |
| 泄漏控制 | 棋盘布局去重，训练集与测试集无布局重叠 |
| Horizon | 12 步；每个选中的棋盘都由精确 BFS 确认为可解 |
| 训练 | 50 个 update，每次 16 条主轨迹 |
| Probe | budget 2，lambda 0.5 |
| Seeds | 17、42、101 |

模型、verl 和数据 revision 记录在 `experiments/results/public_sokoban_main_v1/metadata.json`。四种方法使用相同公开划分和主 rollout 预算。原始轨迹包括运行 manifest、解析后的 Hydra 配置、trainer 日志、标准 rollout JSONL 及每条轨迹的 episode sidecar。

## 4. 结果

下表是三个 seed 的均值和样本标准差。

| 方法 | 最终成功率 | 相对 GRPO 的配对提升 | 额外 rollout token | 有效 probe | 每千 token 高影响 anchor |
|---|---:|---:|---:|---:|---:|
| GRPO | 24.0% ± 2.0% | - | 0.0% | 0/0 | - |
| Random-B2 | 29.2% ± 4.6% | +5.2 ± 6.5 个百分点 | 51.2% | 1140/1200 | 11.93 |
| Surprisal-B2 | 31.2% ± 3.4% | +7.3 ± 4.3 个百分点 | 57.3% | 1142/1200 | 10.44 |
| LinearUCB-B2 | 31.0% ± 1.2% | +7.0 ± 0.8 个百分点 | 51.6% | 1138/1200 | 11.32 |

LinearUCB 在三个配对种子上都高于 GRPO：测试成功数分别多 9、10、8 道（每个种子共 128 道）。Random 在一个 seed 上提升很大，一个 seed 上小幅提升，另一个持平，因此整体波动更高。Surprisal 的数值均值高 0.2 个百分点，但与 seed 波动相比很小，且 rollout 成本比 LinearUCB 高 5.6 个百分点。

调度器诊断不支持原先“LinearUCB 每单位 probe token 找到更多高影响 anchor”的假设。Random 平均为 `11.93`，LinearUCB 为 `11.32`。一种可能解释是二元 high-impact 计数没有考虑 delta 符号、训练相关性和 turn 位置，而 LinearUCB 的 credit 更稳定；但目前这仍是待检验解释，不能当作因果结论。

## 5. 工程故障与修复

完成实验前，团队定位并记录了以下问题：

- AutoDL 无卡模式下 Hugging Face 下载失败；启用网络加速、禁用 Xet 并使用持久化缓存后，下载可恢复。
- 全词表 entropy 计算曾触发 11.97 GiB 显存申请并耗尽显存；分块计算、限制 token batch 和缩短 response 上限后恢复。
- vLLM 与 FSDP actor 同机预留显存导致权重同步期间 actor 被杀；将 vLLM utilization 从 0.45 降至 0.25 后恢复余量。
- verl 因 PPO mini-batch 倍数不兼容，将 16 条真实轨迹补齐到 32；最终将 mini-batch size 设为 4，正好对应 16 行，避免生成没有 sidecar 的伪训练行。
- 一个失败的 Random-B2 目录仍被保留；汇总器只排除没有达到声明最终 step 的运行。

完整时间线和命令见 `experiments/environment/2026-09-21-public-ablation-incidents.md`。

## 6. 局限

1. 三个 seed 足以作为作品集中的重复性记录，但不足以作正式显著性声明。
2. LinearUCB 的 rollout token 成本比原先 `1.5x` 上限高 1.7 个百分点。
3. 当前 probe budget 先按 episode 选样本，还不是全组 top-B。
4. Sokoban 使用二元终局奖励，因此很多 counterfactual 分支结果相同，调度器目标较粗。
5. 前缀重放依赖确定性和可枚举动作；迁移到浏览器环境还需要控制 session 和隐藏状态。
6. 一个事实动作与一个替代动作的局部差值不是该 turn 的完整因果贡献。
7. 保留的 seed 17 GRPO 运行早于最终 PPO mini-batch padding 修正。虽然合成行的 loss mask 为零，但论文级配置一致性仍需重跑该 baseline。

## 7. 复现与作品集使用

运行 `make check` 可检查 CPU 正确性；运行 `make results` 可重新生成已提交的汇总与矢量图。正式 GPU 实验由 `scripts/run_sokoban_ablation.sh` 启动，具体命令见 `experiments/README.md`。

目前最稳妥的简历描述应强调 LinearUCB 配对结果的稳定性，而不是调度器优越性：

> 在当前 verl 上构建了 ProbeGRPO Qwen3.5 Agent-RL 系统，支持确定性 counterfactual 重放、预算感知 anchor 选点和 turn-local advantage shaping；公开 Sokoban 三种子成功率从 24.0% ± 2.0% 提升到 31.0% ± 1.2%，额外 rollout token 成本为 1.516 倍。
