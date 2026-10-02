# 系统架构

## 设计目标

ProbeGRPO 将研究想法与训练框架解耦。无需导入 torch，就能测试重放、选点、probe 和 advantage shaping。当前 verl 只在外围作为适配器接入，不是核心包依赖。RAGEN 仅作为环境实现参考，不使用其训练依赖栈。

## 数据流

1. 标准 Agent-RL runner 为一个任务组生成 `K` 条轨迹。
2. 将框架历史转换为经过校验的 `TrajectoryTrace`，再将每个 assistant turn 转成 `TurnRecord`。其中 `action_prefix` 不包括当前动作；`token_indices` 只覆盖当前 assistant turn 生成的 token。
3. `AnchorScheduler` 在任务组中最多选取 `budget` 个符合条件的 turn。
4. `CounterfactualProber` 创建两个全新环境，并在两边重放相同前缀。
5. 执行不同的强制动作前，校验两边 state hash 和 observation 是否一致。
6. 两个分支使用相同的后缀采样 seed，生成 `ProbeResult`。
7. 调度器观察 `abs(delta) / additional_rollout_tokens`。
8. 将稀疏 credit 打包为 `[batch, anchors, tokens]` mask，并对齐 delta/valid 矩阵。
9. `blend_probe_advantages` 将有效 delta 除以 `max(1, 最大绝对有效 delta)`，保留零值和正负符号、不放大小差值，然后只在所选 token 索引上加 credit。

## 对外契约

### `ReplayableEnv`

```python
reset(task_id: str, seed: int) -> ReplayState
step(action: str) -> ReplayState
replay(task_id: str, seed: int, action_prefix: Sequence[str]) -> ReplayState
```

环境重置必须是确定性的。重放通过重置加执行公开动作实现，避免复制不透明的浏览器/数据库对象。

### `TurnRecord`

核心不变量为：

```text
replay(task_id, seed, action_prefix).state_hash == state_hash
```

如果不满足，probe 无效，不能产生局部 credit。

### Torch/verl batch 字段

```text
advantages        float [batch, tokens]
probe_turn_masks  bool  [batch, anchors, tokens]
probe_deltas      float [batch, anchors]
probe_valid       bool  [batch, anchors]
```

适配器只修改 advantage，不改轨迹奖励或无关 turn。

## 调度器行为

- `random`：成本匹配的随机对照。
- `surprisal`：使用所选动作 token 的负 log probability，作为不确定性启发信号。
- `linear_ucb`：预测每个额外 rollout token 能带来的 credit 绝对值，并使用不确定性 bonus。

LinearUCB 使用七个有界特征：entropy、log-probability margin 的倒数、归一化 turn 位置、此前非法动作比例、合法动作数量、后缀成本和最终奖励。最初 20 个 update 或最初 200 个有效 probe 使用按位置分层的随机选择。

## 失败处理

- 没有替代动作：跳过。
- 事实动作或替代动作非法：跳过。
- 前缀重放失败：跳过，并在 `skipped_reason` 中记录异常类别。
- 状态不一致：严格模式下跳过。
- 没有有效 probe delta：返回原始 GRPO advantage。
- 单个 probe delta 为零：该 anchor 不增加 credit，即使其他 delta 非零。
- 所有 probe delta 均为零：保留标准 GRPO advantage。
- budget 为零：精确回退到 GRPO。

## 当前实现边界

已实现核心 pipeline、当前 verl v1 advantage hook、Qwen3.5 Sokoban AgentLoop、确定性重放、paired suffix 生成、三个调度器、成本统计和公开数据实验启动器。三种子、50 update 对比结果保存在 `experiments/results/public_sokoban_main_v1`。

下一步方法边界是实现真正的 group-wide top-B 选点：当前做法先选 `B` 条采样 episode，再在每条里选一个 turn。WebShop 支持和全量 anchor oracle 相关性仍是作品集扩展，不是已完成功能。
