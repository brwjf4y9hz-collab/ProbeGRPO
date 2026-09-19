# 学习与面试讲解笔记

## 这个项目解决什么问题

普通 GRPO 将整条轨迹的最终奖励分给轨迹中的所有输出 token。长程 Agent 中，这会把成功
错误地归因给无关步骤，也会把失败惩罚给原本正确的步骤。ProbeGRPO 不尝试给每一步都做
昂贵评估，而是在固定额外预算下选择少数步骤进行反事实重放。

## 面试时应讲清楚的四点

1. **为什么必须重放前缀**：只有让 factual 和 counterfactual 从同一状态出发，奖励差才有
   局部归因意义。
2. **为什么需要状态哈希**：浏览器或游戏环境可能因为 seed、session 或隐藏状态不一致而
   无法精确复现；不检查就会把环境漂移当成动作贡献。
3. **为什么 LinearUCB 而不是神经网络**：probe 样本早期很少，特征只有七维；线性模型更
   可解释，也能通过置信上界显式探索。
4. **为什么记录 rollout token 成本**：方法增加了采样，单看成功率会产生不公平比较；必须
   报告在相同训练/采样预算下的收益。

## 当前阶段不要声称的内容

- 不声称首次提出反事实 credit assignment。
- 没跑完 GPU 实验前，不声称提升成功率。
- Tiny Sokoban smoke test 证明的是软件逻辑，不是模型效果。
- Qwen3.5 配置存在不等于旧版 RAGEN 依赖已经兼容。

因此训练栈采用当前 verl 的固定 commit 和 `uv.lock`；RAGEN 只作为 Sokoban/WebShop 环境
实现的参考。先用 GSM8K 跑五个标准 GRPO update，是为了把模型、vLLM、LoRA、FSDP2、
checkpoint 与恢复训练单独验证，再引入多轮环境和反事实 probe。

## 五个核心模块的个人理解

- `replay.py`：根据相同 task、seed 和历史 action prefix，确定性重建当前 turn 之前的
  环境状态。状态被序列化并计算 SHA-256，防止把环境漂移误认为动作贡献。
- `probing.py`：从同一个 anchor 状态出发，分别执行事实动作和替代动作，再计算
  `delta = factual_reward - counterfactual_reward`。
- `pipeline.py`：调度 anchor、运行 probe，并用 `trajectory_to_sample` 把轨迹 ID 映射到
  trainer batch 的行号。
- `advantage.py`：把 probe 得到的 reward delta 标准化为局部 credit，只加到对应
  assistant turn 的 token 上。
- `schedulers.py`：决定有限 probe 预算应该花在哪些 turn 上；LinearUCB 同时考虑预测价值
  和探索不确定性。

这些解释放在学习笔记而非生产源码中，使代码保持英文和标准格式，同时保留自己的理解。

## CPU 数据桥

`rollout.py` 把框架无关的 `TrajectoryTrace` 转成 `TurnRecord`。其中 `action_prefix` 只包含
当前动作之前的动作，因为 probe 必须先回到“做当前决策之前”的状态。

`batching.py` 把稀疏 `ProbeCredit` 打包为：

```text
probe_turn_masks  [N, A, T]
probe_deltas      [N, A]
probe_valid       [N, A]
```

`N` 是 batch 轨迹数，`A` 是每条轨迹的 anchor slot 数，`T` 是 padding 后 token 数。
turn mask 的作用是保证局部 credit 不会传播到提示词、其他 turn 或其他轨迹。

## 建议的项目展示顺序

1. 先展示失败轨迹和 GRPO 的粗粒度 advantage。
2. 点击一个 anchor，展示相同状态下的两条分支。
3. 展示 delta 只写入对应 assistant turn 的 token mask。
4. 对比 random、entropy、LinearUCB 在相同额外 token 预算下选中的 anchor。
5. 最后展示 WebShop 成功率、成本和失败案例，而不是只展示最好的一条轨迹。

## GPU gate 后：从单轮回答到真实环境循环

这次排查发现：固定版本 verl 会把 `ppo_mini_batch_size` 再乘以 `rollout.n`。
2 道题各采样 4 次是 8 条真实轨迹；此前 mini-batch=8 要求对齐到 32，额外 24 条是
loss mask 为零的 padding，并不是更多模型采样。现在 mini-batch=2。

`agent_episode.py` 负责循环，`envs/sokoban.py` 负责游戏规则，
`integration/sokoban_agent_loop.py` 负责与真实模型及 verl 对接。
环境是三个手工二维关卡，接入验证通过后还需要扩展关卡，不能当成正式 benchmark。

先运行 `make agent-smoke`。这是脚本动作，不会调用模型。它在同一个 episode 中先走 up，
收到新棋盘，再走 up 把箱子推到目标。第二次动作的 prefix 是 `(up,)`。

这里 T 不只是模型生成的 token 数：中间 observation 和模板边界也占 response 位置，
但它们的 mask 必须是 0。最终一步成功时，记录的 anchor 状态仍然是执行动作前的非终止
状态；`done_after=True` 不能被误用为排除该 anchor 的 `terminal=True`。

小练习：运行演示，找到第二个 turn 的 token_indices。解释为什么它们没有紧挨着第一个
turn 的 token_indices。然后把脚本动作从 `[up, up]` 改成 `[down, up]`，预测为什么第一个
动作无效却仍然改变状态哈希。不要把测试里的虚构 entropy 当作模型真实不确定性。

真实模型接口目前只提供采样 token 的 log-prob；它不等于分布 entropy。缺失统计量记录为
null，后续从 actor 计算中获取后才能接回 entropy/LinearUCB scheduler。
