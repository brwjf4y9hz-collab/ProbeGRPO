# ProbeGRPO 项目记忆

## 代码与事实来源

- 后续任务以当前仓库代码、测试、配置和已提交的实验记录为事实来源；本文只保存稳定方向，若与仓库内证据冲突，以仓库为准。
- 上游仓库：`https://github.com/brwjf4y9hz-collab/ProbeGRPO.git`；本轮发布整合基线为 `main@67e8118`，保留其较旧本地基线727606d之后的12个提交。
- 优先阅读 `README.md`、`REPORT.md`、`docs/ARCHITECTURE.md` 及对应 runbook；不假设项目仍处于早期原型阶段。
- 当前状态：CPU 正确性、GPU/verl 链路、Sokoban AgentLoop 和三种子公开 Sokoban 对比已完成；已报告 LinearUCB-B2 相对 GRPO 的配对提升及 rollout-token 成本。该结果是可复现的工程结果，不宣称统计显著性或 counterfactual credit assignment 的原创性。

## 目标

围绕 ProbeGRPO 收口，不再横向更换框架。在 RAGEN/verl 的 GRPO baseline 上加入 counterfactual probing：定位关键 assistant turn，通过 factual/counterfactual suffix rollout 得到 `ΔReward`，生成 turn-local credit，再融入原 GRPO advantage。目标是验证它能否改善 credit assignment，进而改善训练效果、样本效率或稳定性。

## 工作原则

- 先完成“训练 → eval → 改一点 → 再 eval”的最小闭环，再谈论文。
- 问题驱动读代码；只追当前链路，不按目录从头通读，暂不深挖 FSDP、Ray、HTTP 等底层细节。
- 只改一个核心变量；每个里程碑都要有可运行命令、验收标准和结果记录。
- 先看失败轨迹再提改动，区分 query、observation 利用、中间 turn、final answer 和 reward 粗糙等问题。

## 学习复现模式

- 用户请求“教我”“带我复现”时，按 `docs/LEARNING_REPRODUCTION_ZH.md` 一次推进一个可验证的检查点。先观察真实输出，再追对应代码，最后请用户用自己的话解释关键机制；不要把助手跑通等同于用户已经掌握。
- 复现使用已安装的 `ai-research-reproduction` 技能核对 README、命令和证据；讲解非平凡机制使用已安装的 `mentor` 技能。技能的自动选择若与本机环境冲突，记录原因并选仓库文档中可运行的最小命令。
- 明确区分 CPU 脚本演示、单次 GPU 训练验证与公开数据正式实验。涉及模型训练时以 GPU runbook 和实际训练日志为准。

## 实施路线

1. 跑通原始 GRPO baseline：`dataset → rollout → reward → advantage → loss → backward/update → validation`。保留 baseline reward/accuracy、training curve、checkpoint 和 rollout trajectories。
2. 追溯 `reward → compute GRPO advantage → DataProto.batch["advantages"] → actor policy loss`，确定 ProbeGRPO 的唯一插入点。
3. 先以 `probe_deltas = 0` 接线，必须满足 `A_new == A_GRPO`。
4. 做 turn-local correctness test：只有 anchor turn 对应 token 的 advantage 被修改，其余 token 不变。
5. 接入真实链路：`trajectory → anchor scheduler → replay → factual/counterfactual rollout → ΔReward → token mask → apply_probe_credits_tensor() → GRPO update`。
6. 小规模比较 GRPO 与 ProbeGRPO：同模型、数据、seed、rollout 数、batch size、learning rate 和 steps；首轮用 Random scheduler、1 个 anchor、固定 budget、`λ=0.5`，只验证是否有 signal。
7. 有趋势后再做失败分析，调整 anchor/budget/λ，最后扩展 main results、ablation、cost/efficiency 和 case study。

## 论文判定标准

论文需完整回答：现有 trajectory-level GRPO 的问题是什么（Problem）；ProbeGRPO 改了什么（Method）；为何有效（Why）；实验是否支持（Evidence）。完成小规模 baseline vs ProbeGRPO 闭环才算完整项目；效果和证据链都成立后才进入论文阶段。

## 当前下一步

早期 baseline 与 ProbeGRPO 主实验已完成，不要无证据地重启早期实施路线。下一步由用户具体任务决定；可选未完成方向包括 trajectory demo 和 WebShop 扩展。任何修改都应先定位对应代码、测试、配置与已有实验证据，然后做最小、可验证的变更。

2026-10-15 补充：真实正/负/零 trajectory 案例和 warm-up 位置审计已补入 v1 结果目录；修复固定桶顺序，warm-up 按实际轨迹长度分桶，UCB horizon-relative 特征不变。v2 对照协议见 `docs/FINAL_CONTROLS_ZH.md`，包括 GRPO-K4、GRPO-K6、UCB probe-only λ=0 和修复 UCB，保持原模型、512/128、50 updates、三种子；GPU 结果待运行，不与 v1 混算。用户明确要求：需要 GPU 就说明，不得因为无卡缩减正式实验规格或规模。

同日后续决定：用户批准统一 200-step 版本。默认新协议为 `experiments/protocols/public_sokoban_controls_v3_200.json`，仍为四组×三个种子、原模型/512–128 split/LR/warm-up；每 10 步验证、每 50 步保存、最多保留两份 LoRA actor checkpoint。主终点 step200，旧 50-step 结果与 v2 准备记录保留不改。当前只有无卡代码/配置验收，无新 GPU 训练结论；不要将 smoke、单种子或 50-step 运行冒充正式 200-step 矩阵。

200-step 部署状态：本地代码和测试已完成；本轮原 SSH 地址连接被关闭，尚未把 200-step 版同步到远端。之前远端 90 项全通过是 v2 准备版，不能冒充 200-step 新版的远端验收。恢复连接后先核对/备份/同步新版，再做无卡验收和有卡 smoke；运行命令显式传入新协议文件，防止误跑旧默认 50 steps。

后续实验准备：保留 `single_48gb` 为默认单卡执行配置，另保留 `single_24gb` 与 `dual_24gb` 候选；用户明确要求单卡版本不能被双卡替换。候选 24GB 方案尚未实测，禁止声称可跑。硬件选择不改正式四组×三种子×200 steps 或模型/数据/全局 batch/序列长度；输出按 profile 隔离，同源码同硬件四组 smoke 后才放行主实验。新增验收/失败分类/指标哈希及硬件混用检查。另按用户六条简历标准维护 `docs/PORTFOLIO_ACCEPTANCE_ZH.md`，公开 clone 验收、提交部署和新 GPU 结果仍待完成。

2026-10-16 求职收口：新增中文复盘、真实调用链、GiGPO定位、面试提纲和隔离clone工具。原SSH已恢复，
在远端独立`outputs/portfolio-prep-20261016-v1/source`完成115项CPU单测全通过；本地隔离快照115项中2项缺torch跳过，
README最小CPU复现证据核验通过。六组GRPO/UCB历史完整训练轨迹状态审计和12组工程日志摘要已补入v1目录。
状态重复充分，不能用“几乎不重复”回避GiGPO；分组审计不是GiGPO训练，最终分数SD不是credit方差或机制证据。
未提交推送、未覆盖原远端工作区或修改安装指向、未跑GPU；200-step与硬件候选仍待验收。详见
`experiments/environment/2026-10-16-portfolio/README.md`，不得把独立源码CPU测试冒充已正式部署或公开clone成功。

同日发布整合：用户已授权`codex/portfolio-controls-200`分支提交/推送/创建PR，不授权合并main。
此版本保留公开main的精确结果数据与旧ZIP，当前成本正文统一约51.6%，旧51.7%只说明舍入口径。
历史准备记录保留当时状态，最新状态以`docs/RELEASE_STATUS_ZH.md`和实际PR/验收为准。
学校服务器已提供配置：四张4090 24GB；空闲不等于获配，密钥登录和home目录仍需处理，具体卡号/时长待用户确认。
尚无学校GPU smoke或200-step结果，不因启动条件变化降低科学规格。

2026-10-17 更新（学校用户日志，不是助手直接登录验收）：模型13文件校验和512/128数据准备完成；
dual_24gb四组一步均有completion receipt；GRPO双rank LoRA/optimizer/RNG/LR scheduler从step1恢复，
新增一步完成step2。正式矩阵seed17 GRPO-K4已有200-step完成回执，但K6随后在dense entropy申请5.77GiB时OOM，
其余组未完成，不能称完整矩阵。probe一步的8个delta全零，零credit符合设计，但非零分支尚未由该运行覆盖。
修复详见docs/DENSE_ENTROPY_FIX_ZH.md：保留固定verl commit和原trainer patch，额外精确引擎补丁令dense分支
真正按token分块，保留dtype、梯度和所有正式科学规格。新源码必须新目录重做四组smoke，不能复制旧回执或自动混合
旧baseline与新矩阵。代码/CPU验证不等于学校GPU长批次通过。学校home缺失应单独配置Triton、FlashInfer等缓存；
不改HOME，不重装依赖，不减rollout/序列/步数/种子。SwanLab按用户要求延期，不要安装或请求API Key。
