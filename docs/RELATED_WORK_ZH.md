# GiGPO 与 ProbeGRPO：为什么要付费重放

核对日期：2026-10-16。依据 [GiGPO 论文 v3](https://arxiv.org/html/2505.10978v3)
第 4 节及 [官方实现](https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py)。
官方 master 是可变版本，正式适配前需固定 commit、超参数和版本差异；当前没有实现或跑过 GiGPO 对照。

## 1. 同一个问题，不同的信息来源

GiGPO 将同一任务组内重复出现的状态聚合，用已有后续回报构造 step-level 相对优势，
再与 episode-level 优势合并。它不需要为此额外生成轨迹，但分组和统计并非字面上的零计算。
ProbeGRPO 则主动重放少量决策点，对原动作和一个合法替代动作重新生成 suffix；
因此可以主动获取比较，也必须承担额外推理成本和采样噪声。

| 比较项 | GiGPO | 当前 ProbeGRPO |
|---|---|---|
| 比较样本 | 主 rollout 已经遇到的同状态样本 | 选定 anchor 后主动生成的一对 suffix |
| 局部统计 | 同状态组内的后续 return 相对值 | factual reward − counterfactual reward |
| 额外 rollout | 分组本身不需要 | v1 UCB 增加约51.6%训练生成token（旧舍入记录51.7%） |
| 信号不足情形 | 状态很少重复、组内回报无差异 | 替代动作无效、重放失败、两分支同回报 |
| 已有本项目实测 | 无 | 三种子 50-step；仍有旧配置和成本对照限制 |

这不是“GiGPO 无用所以重放更好”。更合理的问题是：**自然状态重用已经能提供局部信号时，
主动重放增加的信息是否值得它的成本？** 本项目目前还不能回答，不能跨模型/环境直接比较论文分数。

## 2. 为什么不能直接把 replay hash 当分组键

`SokobanEnv.state()` 的 hash 包含 task、seed、布局、玩家/箱子位置、steps、max_steps 和 invalid_actions，
目的是严格验证重放一致性。相同棋盘在不同剩余步数下，可能有不同的可达回报；随便删除这些字段
会把不同有限时域状态合并。另一方面，严格 hash 也可能因与分组目的无关的计数而过度拆组。

模型还保留累积 action/observation 历史。同棋盘不代表同策略输入，不能把“环境等价”自动解释为
“模型条件完全相同”。官方实现按任务组内 anchor observation 分组，我们的 history-concatenated
AgentLoop 与其 step-independent 组织方式不同；迁移时必须明确这是何种适配。

## 3. 无卡先检查信号机会，不伪造算法成绩

`scripts/audit_state_groups.py` 对完整历史 sidecars 做四种诊断：严格 replay hash、棋盘 observation、
observation + 剩余动作数、已记录文本历史。只在同一 run、同一 update、同一原始 prompt UID 内分组；
不把碰巧抽到同一棋盘的不同 prompt 实例拼起来，不混 train/test 或不同策略更新。

输出组大小、重复 occurrence 比、跨轨迹复用比、回报异质性、动作与回报都有差异的跨轨迹比例；
单列非初始动作位置，避免所有 K 条轨迹共享初始状态造成虚高。return 明确使用 gamma=1，
是稀疏二值 Sokoban 诊断，不冒充论文超参数。回报异质性只是信号机会，不证明动作因果贡献。

```bash
python3 scripts/audit_state_groups.py /path/to/complete/run \
  --rollouts 4 --prompts 4 --output-dir outputs/state-groups-new
```

脚本核对训练步数及每个 UID 的 K 条 session 完整性，缺原始 sidecar、缺 rollout、重复记录时拒绝汇总。
`groups.csv` 和带来源哈希的 `summary.json` 供复核；精选三个 trajectory demo 不能替代总体审计。
真实审计记录见 `experiments/results/public_sokoban_main_v1/state_group_audit/`（若尚无完整结果，README 会明确说明）。

## 4. 正式对照的边界

当前批准的 v3 仍是四组 × 三种子 × 200 updates，不默默加入第五组。正式 GiGPO 适配需先固定
状态定义、discount、归一化、局部权重、singleton/零方差处理、token mask，以及论文/代码版本差异。
测试至少包括跨任务隔离、全零局部信号回退、仅 assistant token 更新和手算小例。

若用户另行批准同规格 GiGPO 三种子，总矩阵才成为 15 runs、3000 updates；每组仍 200 步。
不需要额外 probe 不意味着这三次训练无需 GPU。也不以删现有对照、缩数据或减种子腾预算。

## 5. 现在可以怎样回答面试官

我没有把 step-level credit 当成自己的原创。GiGPO 复用自然重复状态，ProbeGRPO 主动采样局部比较；
后者值不值额外 token，是需要实验回答的问题。旧结果显示正向信号，但没有证明等成本优势，
也没有证明 LinearUCB 的收益来自 credit 稳定性。我的工作价值首先是把重放、token 对齐和训练接线
做成可检查的系统，并把这些假设变成可运行、可报告负结果的对照。
