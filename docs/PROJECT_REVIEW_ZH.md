# ProbeGRPO 复盘：从整条轨迹的奖励，到一个动作的训练信号

仓库：[ProbeGRPO](https://github.com/brwjf4y9hz-collab/ProbeGRPO)。状态截至 2026-10-16：
真实完成的是三种子 50-step 比较；修复后的 200-step 对照仍待 GPU。本文沿着“信号为什么粗、
怎样接进训练、证据在哪里、哪些解释还不成立”展开，不把待跑实验写成成绩。

## 1. 我想解决的不是让助手会聊天，而是让奖励能指出动作问题

Sokoban 的任务很具体：模型读取棋盘，每次只能选择 up/down/left/right，将箱子推到目标位置。
正式数据来自固定版本的公开 RAGEN 数据，512 个训练棋盘、128 个 held-out 棋盘，布局去重且无交叠，
每个都用 BFS 检查在 12 步内可解。因此失败不是靠模糊的语言 judge 判定，终局奖励是成功 1、失败 0。

但整条轨迹的 0 或 1 不告诉我们哪一步做错了。GRPO 能比较同一题的多条轨迹，却不能仅凭该分数
区分一个正确早期动作和后面的失误。我的目标不是重新实现 GRPO，而是在已有训练链路中补一个
局部比较，并严格限制它修改的 token 范围。

## 2. 先验证同一个状态，再比较两个动作

一个 probe 先重放 anchor 之前的动作，校验完整 state hash 和 observation。原动作与替代动作
从相同环境状态分别出发，并重新生成后续动作；用 factual 减 counterfactual 的最终回报作为 delta。
这里 factual 也重新采样，不能把它误读为原主轨迹必然得到的回报。

真实正例在 `trajectory_demo/README.md`：anchor 前玩家紧挨箱子，原动作 right 将箱子推到目标，
替代 left 则离开；保存的分支回报为 1/0、delta=+1，额外生成 20 tokens。仅 assistant response
索引 `[61, 62]` 接收局部修正。λ=0.5 时，按公式的局部增量是 +0.5；这不是恢复出来的 actor 梯度。

这份案例保存了主轨迹和 probe 回报，CPU 独立重放了 anchor 和单动作转移；两条完整 suffix
当时没有保存，所以我不展示编造的后续路径，也不声称重新验证了 suffix 的模型输出。

## 3. 真正困难的是不把信号接错位置

多轮序列包含 assistant 动作和环境 observation；它们都可能占据 response segment 的位置，
但只有 assistant tokens 应参与动作的 policy loss。`TokenStream` 校验合并前后的 token 前缀和 mask，
`EpisodeTurn.token_indices` 记录动作对应的位置，sidecar 再按 trainer 的 UID/session 对齐批次。

唯一接入点在 verl 计算完 GRPO advantage 之后、写回 TransferQueue 之前：
`A_new = A_GRPO + λ × normalize(delta) × turn_mask`。训练器后面的 policy loss、backward 和
optimizer update 仍交给上游。budget=0、λ=0、无有效 probe 或 delta=0 时，张量级接线必须回退，
不能靠“训练还能跑”来判断正确。

测试与 CPU smoke 用于验证这些契约，真实 GPU 日志才用于证明模型更新。两者不能互相替代。

## 4. 三次故障怎样改变了执行方案

### 4.1 entropy OOM：先确定失败申请，不虚构峰值优化

日志记录 full-vocabulary entropy 试图额外申请 11.97 GiB，而当时只剩 7.17 GiB。
我启用了 chunk size 256、actor token budget 2048，并统一 response cap 1024 等配置后继续运行。
这组调整解决了当前设置的执行问题，但不是单变量性能实验；11.97 GiB 不是原峰值，不能写成
“峰值降低 X%”或把效果全部归因于一个开关。

### 4.2 vLLM/FSDP 共置：预留不是实际峰值

同步权重时 actor 退出，Ray 报 EOF，没有正常 Python 异常。事故记录将其定位为显存瞬时余量不足，
随后把 vLLM utilization 从 0.45 调到 0.25，下一次通过了初始化和初始验证。
这是配置与症状之间的可核验记录，不是 profiler 级的 OOM 因果证明，也没有测得独立吞吐收益。

### 4.3 16 条真轨迹被补到 32 行：不能给不存在的轨迹补证据

旧 PPO mini-batch=8 导致 trainer 要求 32 的倍数，16 条真实轨迹被 padding，合成行没有环境 sidecar，
于是 join 正确地失败。修改 mini-batch=4 后兼容 16 条真轨迹，而不是为合成行伪造 probe。
seed17 的 GRPO 已在修改前完成，因此旧比较保留配置差异；新版完整重跑基线来消除它。

以上记录来源于 `experiments/environment/2026-09-21-public-ablation-incidents.md`。
如果要声称峰值/吞吐提升，还需同模型、长度和 batch 下的匹配测量，当前不填未知的 X、A、B。

## 5. 正向结果与反直觉结果必须一起讲

四方法 × 三种子 × 50 updates 的结果中，GRPO 成功率为 24.0±2.0%，UCB 为 31.0±1.2%；
配对提升7.0±0.8个百分点，精确记录的额外训练生成token约51.6%（旧舍入汇总为51.7%）。±是三个种子的样本SD，不是置信区间。
Surprisal 均值 31.2%，不能因为 UCB 更稳就说 UCB 均值最好。

Random 每 1000 probe token 找到的 high-impact anchor 为 11.93，高于 UCB 的 11.32。
这否定了我用该指标证明调度效率最优的说法，但不证明调度无作用，也不证明 credit 稳定性是原因。
最终分数的种子波动和 credit 的估计方差不是同一个量。

后续审计又发现旧 warm-up 的固定桶顺序造成前段偏置：612 个选中 anchor 中 610 个在 horizon
前段，且三个种子直到 step28 才首次进入真实排名。当前代码修复分桶与采样顺序，旧结果不反向更改。
50 步只有 200 次 prompt 暴露，约 0.39 个数据集遍历量，也不足以讨论收敛后的表现。

## 6. 下一步不是换环境，而是把解释拆成能失败的对照

新协议保留四组：GRPO-K4、GRPO-K6、有 probe 但 λ=0、完整方法 λ=0.5；三个种子各 200 步。
它们分别检查主采样量、执行 probe 本身和 credit 注入的影响。K6 既改变成本也改变组内估计，
必须报告实际 token/时间，不能先称等算力。主终点固定 step200，其他点是预定动态诊断。

GiGPO 是必须讨论的近邻方法：复用已有同状态轨迹，不需要额外 probe。我们尚未完成其适配与对照，
现在先做完整轨迹上的状态重用审计；方法差异和未来条件见 `RELATED_WORK_ZH.md`。
本项目不以 counterfactual credit 的原创性、SOTA 或已证实的稳定性机制作为求职卖点。

## 7. 我能负责的边界

本仓库的核心是 replay contract、Sokoban adapter、paired suffix probing、三种 scheduler、
turn-local mask 和 verl sidecar 接线，以及配置/成本/来源验收。模型、vLLM 推理、FSDP2 分片、
Ray 和优化器主体来自上游。把这些边界说清楚，比把整个训练框架都归到自己名下更重要。

48GB 单卡默认入口继续保留；24GB 单卡/双卡只是待测执行候选，不因无卡缩短数据和实验。

## 待补充 / 待核实

- 当前源码的隔离 clone 验证与公开远端版本验收要分别记录；本地成功不能冒充已发布。
- 修复后的正式 12 组 × 200 steps、保存/恢复验证，以及学校硬件显存/速度。
- GiGPO 分组定义与正式对照；若扩矩阵另行确认预算，不偷偷删原对照。
- 若主张 credit 稳定性机制，需固定策略/anchor 的重复采样与匹配训练消融。
- 两条完整 probe suffix 轨迹、匹配工程性能测量仍缺；不能用精选例子补造。
