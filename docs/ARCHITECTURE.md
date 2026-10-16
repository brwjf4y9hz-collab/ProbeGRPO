# 系统架构

## 设计目标

ProbeGRPO 将研究想法与训练框架解耦。无需导入 torch，就能测试重放、选点、probe 和 advantage shaping。当前 verl 只在外围作为适配器接入，不是核心包依赖。RAGEN 仅作为环境实现参考，不使用其训练依赖栈。

## 数据流

1. 标准 Agent-RL runner 为一个任务组生成 `K` 条轨迹。
2. 将框架历史转换为经过校验的 `TrajectoryTrace`，再将每个 assistant turn 转成 `TurnRecord`。其中 `action_prefix` 不包括当前动作；`token_indices` 只覆盖当前 assistant turn 生成的 token。
3. GPU接入先选每个prompt组的前`B`条episode，再由`AnchorScheduler`在每条选一个turn。通用核心支持组级候选，但不是当前GPU预算策略。
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

- `random`：相同probe数量的随机对照，不保证实际生成token成本相同。
- `surprisal`：使用所选动作 token 的负 log probability，作为不确定性启发信号。
- `linear_ucb`：预测每个额外 rollout token 能带来的 credit 绝对值，并使用不确定性 bonus。

LinearUCB 使用七个有界特征：entropy、log-probability margin 的倒数、归一化 turn 位置、此前非法动作比例、合法动作数量、后缀成本和最终奖励。最初 20 个 update 或最初 200 个有效 probe 使用按位置分层的随机选择。

当前warm-up每轮打乱非空位置桶，按实际轨迹长度分桶，避免budget=1总选最早非空桶。
UCB的`turn_fraction`特征仍按horizon定义，新增`trajectory_fraction`只服务于warm-up。
20 updates和200个有效且有成本probe两个条件均满足后才排名。旧v1的偏置不因修复而消失，
见[200-step对照](FINAL_CONTROLS_ZH.md)和[位置审计](../experiments/results/public_sokoban_main_v1/anchor_audit/README.md)。

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

## 实际GPU调用链与实现归属

```text
run_final_controls.py                  固定v3矩阵、硬件与smoke门槛
 -> run_public_sokoban_train.sh        数据/模型路径、固定verl patch和实际参数manifest
 -> SokobanAgentLoop.run()             configs/agent_loop/sokoban.yaml注册
      -> run_episode()                token stream、动作/观察交互
      -> select_episode_anchor()      前B条episode中各选一个anchor
      -> probe_episode()              prefix重放、校验、成对suffix与delta
      -> episodes/step-N/*.json        UID/session、assistant indices、probe记录
 -> verl原始GRPO advantage
 -> apply_sidecar_probe_credits()      严格join和mask校验
      -> apply_probe_credits_tensor()  仅局部相加
 -> TransferQueue -> 上游actor policy loss -> backward -> optimizer step
 -> 定期held-out评估                   评估不注入probe credit
```

固定patch与`install_verl_v1_probe_hook.py`将接入点限定在GRPO advantage之后、TransferQueue写回之前。
`pipeline.py`是可复用CPU核心，不是GPU AgentLoop的启动入口。模型、Ray、vLLM、FSDP与优化器主体归上游；
本仓库负责重放/probing、调度、mask、sidecar和验收工具。

模型只输出移动动作，没有shell或任意文件写入工具；非法动作消耗尝试次数。训练Python进程不具备OS沙盒，
可见卡检查也不替代学校调度权限。GiGPO状态复用与策略历史的区别见[相关工作](RELATED_WORK_ZH.md)。
