# 面试讲解提纲：可以讲到哪里，不能讲到哪里

配套阅读：`PROJECT_REVIEW_ZH.md`、`ARCHITECTURE.md`、`RELATED_WORK_ZH.md`。
这里提供的是准备提纲，不代表使用者已经掌握代码；讲不清的节点应回到对应测试和真实轨迹。

## 开场约 30 秒

ProbeGRPO 是一个多轮 Agent 后训练项目。我研究终局奖励过粗时，能否通过确定性重放，比较原动作和
替代动作的后续回报，再把差值只接到这个 assistant turn 的 advantage。底层用 verl/vLLM/FSDP，
我做的是环境重放、分支 probing、调度、token mask 和训练接线。旧三种子50步实验中，UCB版本成功率
从基线24.0%到31.0%，但多用了约51.6%的训练生成token；成本归因和修复后的200步对照还没完成。

## 架构按一条真实轨迹讲，不按目录背

1. 模型看到棋盘，生成动作，环境返回新棋盘；记录每个 assistant turn 的 token 索引。
2. 从被选 episode 中选一个 anchor，重放 prefix，严格检查 hash/observation。
3. 原动作和一个合法替代动作分别生成 suffix，得到 factual−counterfactual reward。
4. sidecar 按 UID/session 对齐 trainer batch，在标准 GRPO advantage 后加局部修正。
5. 上游 actor 用修正后的 advantage 计算 policy loss、反向传播和更新；验证不注入 probe credit。

正例用真实 `[61,62]` token 的案例；强调未保存完整 suffix，不展示想象路径。详见
`experiments/results/public_sokoban_main_v1/trajectory_demo/README.md`。

## 必须能回答的追问

**为什么用 reward 差而不是直接改整条轨迹 reward？**
目标是只修正局部动作，直接改整条轨迹会重新把信号摊到其他动作；mask边界由测试检查。

**和 GiGPO 有什么不同？**
GiGPO复用已有轨迹的同状态组；这里主动采样一个替代动作，付出额外成本。
旧轨迹确实存在大量状态重用，因此不能用“状态不重复”回避对照，目前不声称比GiGPO好。

**+7pp是不是多花算力换来的？**
当前无法排除。新版准备了K6和λ=0对照；K6仍不是自动等算力，需要实际token/时间测量。

**为什么说UCB更稳定？**
只能说三个种子的最终成功率SD较小；不等于credit估计更稳，也未证明收益机制。
Random的high-impact/token指标反而更高，这个反例必须一起讲。

**你处理过最难的问题是什么？**
选一个讲透：entropy申请11.97GiB失败；vLLM预留与FSDP同步冲突；或16条真轨迹被padding到32行。
说明证据、改法、验证和未知项，不把失败申请当峰值，也不编性能提升百分比。

**哪些是你写的，哪些是上游？**
本仓库负责replay/probing/scheduler/mask/sidecar和验收工具；模型、推理引擎、分布式训练和优化器主体是上游。
代码行数不是主要贡献证据，测试契约、故障定位和实验边界才是。

## 简历表述边界

可以写：基于verl实现多轮Agent反事实重放与turn-local advantage，在公开Sokoban旧三种子50-step
比较中观测到约+7pp配对提升，同时报告约1.516×训练生成token成本，并实现轨迹追踪和正确性验收。

暂时不写：首次提出counterfactual credit、SOTA、显著提升、等算力领先GiGPO、通过稳定credit解释收益、
24GB显存已适配、完整200步实验已完成。新版公开可复现结论也要等目标commit推送后重新clone验证。
