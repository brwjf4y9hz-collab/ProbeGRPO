# 项目问题讲述与追问

讲述时按“具体现象 → 怎么确认 → 为什么选这个改法 → 怎样验收 → 什么仍不知道”展开。以下是基于证据的讨论提纲，不表示已经评估本人对实现的掌握程度。

## P03：零 delta 被中心化成负 credit

最早两个有效 probe 的原始 delta 为 [1.0, 0.0]，中心化 z-score 却得到 [+1, -1]，零差值也改变 assistant turn 的 advantage。

改为除以 max(1, 最大绝对有效 delta)，不减均值，只将 credit 加到对应 assistant turn mask；budget=0、lambda=0 与 zero delta 分别保留基线行为检查。

当前实现将 [1,0] 映射为 [1,0]，零 delta 仍为零；相应代码与后续主实验结果封存，首次 z-score gate 不作为最终性能对照。

追问准备：能否打开 [P03 证据链](RETROSPECTIVE.md#p03)解释判据；哪一步是日志直接观测，哪一步只是判断；如果再次失败，先核对哪份输入/状态/配置？

不能省略的限制：局部单替代动作的奖励差仍不是完整因果贡献；通过数值不变式也不能单独证明最终任务收益。

## P05：合成 padding 行没有真实 episode sidecar

Random-B2 将 16 条真实轨迹补到 32 行后，在读取合成行 probe sidecar 时触发 FileNotFoundError。

mini-batch 改为 4，使要求倍数匹配 16 条真实 rollout；不伪造 sidecar 或 credit。失败目录保留 .failed-padding，汇总器拒绝没有达到声明最终 step 的运行。

修正后 probe 臂完成 50 updates；最终三种子统计明确列出 seed17 GRPO 早于 padding 修正的历史差异。

追问准备：能否打开 [P05 证据链](RETROSPECTIVE.md#p05)解释判据；哪一步是日志直接观测，哪一步只是判断；如果再次失败，先核对哪份输入/状态/配置？

不能省略的限制：seed17 GRPO 未以修正后配置补跑；零 loss mask 不足以消除协议差异。论文级因果对照仍需消除该差异。

## P07：成功率提高，但效率目标没有全面达到

LinearUCB 成功率高于 GRPO，但高影响 anchor/千额外 token 没有超过 Random；总 rollout token 比例 1.516×，略高于原 1.5× 目标。

同时报告四方法、三个种子、配对差值、样本 SD、token 与 GPU 时长；结果图保留 Random/Surprisal 的优势项，不只挑最好看的 LinearUCB 指标。

LinearUCB 30.99% 对 GRPO23.96%，配对均值+7.03pp，额外token51.64%；Random 的诊断11.93高于LinearUCB11.32，Surprisal的成功率均值31.25%更高。

追问准备：能否打开 [P07 证据链](RETROSPECTIVE.md#p07)解释判据；哪一步是日志直接观测，哪一步只是判断；如果再次失败，先核对哪份输入/状态/配置？

不能省略的限制：三种子不证明显著性；历史配置差异仍在。不能声称最优调度、普遍稳定、严格等成本收益或完整因果贡献。

## P08：归档能重算结果，却没有主实验最终模型

主实验关闭 checkpoint 保存；现存大型 checkpoint 是早期 GSM8K step6。数据归档第一次整包核验因额外 metadata 文件拒绝，旧四舍五入 CSV 的 token 开销比例显示也偏粗。

按清单核验实验、模型和数据备份，单独核验解压数据目录；从原精度重建 CSV并保存legacy版。将代码、固定框架 patch、原始证据、模型/数据、checkpoint 分开归档。

历史核验记录：12次运行1536条评估、128共同棋盘、50步；20,399证据条目与20,412完整备份条目匹配；token 开销比例显示由51.7%修正为51.6%，成功次数不变。

追问准备：能否打开 [P08 证据链](RETROSPECTIVE.md#p08)解释判据；哪一步是日志直接观测，哪一步只是判断；如果再次失败，先核对哪份输入/状态/配置？

不能省略的限制：最终 Sokoban 权重仍不存在，GPU 冷启动重训未执行。公开档案复核结果不等于从头训练复现，历史服务器清单不等于本轮重新访问GPU。

## 避免说过头

不要把“同B=2”说成等总token；不要把chosen-token surprisal说成完整词表entropy；不要把早期GSM8K checkpoint说成主实验模型；不要把历史GPU记录说成本轮重新训练。
