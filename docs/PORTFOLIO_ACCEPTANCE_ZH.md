# ProbeGRPO：简历级项目验收与证据索引

这份清单采用用户提出的六条标准。它是**证据与缺口索引，不是“全部已完成”的自评证明**。
ProbeGRPO 定位为多轮 Agent 后训练/credit assignment 研究工程项目，不是通用聊天助手或已上线业务产品。

## 1. 明确用户和任务

目标使用者：需要在有限 rollout 预算下研究多轮决策信用分配的后训练研究工程师。
具体任务：Qwen 在 Sokoban 棋盘上多轮选择移动动作，将箱子推到目标位置；环境提供稀疏成功回报。
研究问题：整条轨迹的 GRPO 信号较粗，能否用可重放状态上的 factual/counterfactual suffix 回报差，
只修正一个关键 assistant turn 的 advantage，并改善成功率、学习动态或稳定性？

这不是“给用户做任意任务”的产品需求。没有声称开展真实用户访谈、生产部署或证明商业价值。

## 2. 可运行入口

根 README 提供 Python 标准库 CPU 入口，无模型、无 API Key：

```bash
PYTHONPATH=src python3 -m probegrpo.smoke
PYTHONPATH=src python3 -m probegrpo.agent_smoke
make check
```

CPU 展示数据流、环境交互、重放和 mask 正确性，不能冒充模型学习。
正式入口是 `scripts/run_final_controls.py` + 固定 `public_sokoban_controls_v3_200.json`；
默认只输出计划，明确 `--execute` 才能训练。环境、缓存和 GPU runbook 分开写，避免 clone 后直接付费排错。
本轮新增内容在`codex/portfolio-controls-200`发布分支整合，提交/推送/公开clone是独立门槛。
具体状态以`docs/RELEASE_STATUS_ZH.md`及实际GitHub PR为准，不能用本清单代替发布证据。
新增 `scripts/check_clean_cpu.py` 和 `docs/RELEASE_CHECK_ZH.md`，将未提交源码快照、已提交本地 clone、
公开 GitHub clone 三种证据分开；只有第三种能证明发布版本可运行。
交付前仍需审查提交必要代码、排除 outputs/密钥，再对公开 commit 复核新入口。

## 3. Agent loop、工具/上下文/权限边界

| 机制 | 代码证据 | 边界 |
|---|---|---|
| loop 注册 | `configs/agent_loop/sokoban.yaml` | 将 `probegrpo_sokoban` 注册到 verl AgentLoop，不是通用工具注册平台 |
| 动作工具契约 | `src/probegrpo/envs/sokoban.py` 的 `ACTIONS`、`step()` | 仅 up/down/left/right；非法文本或堵塞动作消耗一次尝试，不执行 Python/shell |
| 多轮 loop | `src/probegrpo/agent_episode.py` 的 `run_episode()` | 观测→模型动作→环境转移→观测，按成功、horizon、token budget 或空生成停止 |
| 上下文管理 | `TokenStream`、`validate_merge()`、`VerlTokenIO` | 区分 prompt、assistant、observation；已记录 assistant token 不允许被后续 merge 改写 |
| 重放/credit | `episode_probing.py`、`integration/verl_v1_sidecar.py` | 同状态/同观察再分支；mask 只覆盖 anchor assistant tokens，非 observation/padding |
| 预算/资源 | `probe_mode.py`、`scripts/control_hardware.py` | B 限制 probed episodes；GPU 仅限明确分配的可见卡，不能自动抢其他卡 |

模型动作接口没有 SSH、任意文件写入、任意网络请求或删除工具。训练 Python 进程仍需要正常的
文件/模型服务/可选 SwanLab 网络权限，**并未实现操作系统级沙盒或完整企业 RBAC**。
学校作业权限由调度器/管理员管理，脚本中的可见卡检查不是权限隔离的替代品。

## 4. 日志与 trace

- 原始 train log：逐 step 的 reward、loss、梯度、时延等。
- `rollouts/episodes/step-N/*.json`：动作前缀、前后观察、完整状态哈希、token mask、停止原因、probe 结果。
- replay 不一致、无候选、非法替代动作、suffix 异常有明确 skip reason；无效 probe 不注入 credit。
- 三个真实正/负/零案例见 `experiments/results/public_sokoban_main_v1/trajectory_demo/`，含来源哈希。
  只重放了状态/单动作转移，原始两条完整 probe suffix 没保存，不展示虚构后续轨迹。
- `anchor_audit/` 给出 warm-up 位置偏差及机会集合，修复前/后不混淆。
- 新版 receipt/summary 单列训练、历次验证、最终验证的失败类型；**该代码已准备，正式 200-step 输出待 GPU 运行**。

## 5. Eval：成功率、失败类型、成本与延迟

| 项目 | 已有证据 | 仍缺什么 |
|---|---|---|
| case 与泄漏控制 | 固定公开 RAGEN revision；512/128 去重，board 零重叠；BFS 可解性检查 | 不是跨环境泛化证据 |
| 成功率 | v1 四方法三种子 50-step 的原始记录/逐步曲线 | 修复后的 200-step 全部四组对照尚未训练 |
| 失败类型 | 轨迹有停止原因、非法动作和 probe skip；新版自动分类验收/汇总 | 新版最终 held-out 失败分布尚无结果 |
| 成本 | 主/额外生成 token、累计更新时间；v1 成本口径已审计 | K6 是否实际近似等 token 要测量；不是已证明等 FLOPs/账单成本 |
| 延迟 | v1 原始 log 和进度条；单步/整轮区分 | 学校 24GB 配置速度/峰值显存/恢复还未验证 |
| 稳定性 | v1 三种子描述性 SD、配对差值 | 不宣称统计显著性、200-step 收敛或 LinearUCB 必然最优 |

v3 主协议：GRPO-K4、GRPO-K6、UCB-B2 λ=0、UCB-B2 λ=0.5，三种子各 200 步；主终点预定 step200。
trace inspector、完成检查与完整矩阵汇总都不能被“只看到最后一行 log”替代；中断结果保留，不自动覆盖。
所有硬件 profile 保持原模型/数据/预算，24GB 候选先经四组 GPU smoke，不以 CPU fixture 代替训练。

## 6. 设计复盘与替代方案

已有复盘见 `REPORT.md`、`docs/ARCHITECTURE.md` 与 `experiments/environment/`。
中文连贯复盘见 `docs/PROJECT_REVIEW_ZH.md`；它保留原始失败、未知工程收益和上游贡献边界。
`engineering_audit/summary.md` 从12组历史 log 提取内存/step时延，带来源哈希，不冒充优化前后消融。

- 为何采用 deterministic replay：需要把动作变化与初始状态差异分开；代价是要求环境可确定性重放。
- 为何用成对 suffix：factual 也重新采样，减轻把原轨迹的后续随机结果直接当唯一参照的问题；
  代价是额外推理 token，且一次替代动作仍不是完整因果贡献。
- 为何 turn-local mask：把信号限定到一个 assistant turn；代价是复杂的 token 对齐和 sidecar join 校验。
- 替代方案包括保留 trajectory-level GRPO、均匀随机 anchor、surprisal anchor、额外主 rollout、
  learned scheduler；前几种已有 v1 比较，额外主 rollout 与 no-credit probe 将由 v3 补证。
- LLM judge/学习过程奖励是其他候选方向，本项目未实现或比较，不能声称它们更差。
- “机制真的提升了吗”：v1 观察到正向配对差值，但有 warm-up 偏置、seed17 配置差异、训练较短与成本对照缺口。
  新版完整结果出来前，只能报告已有现象和待验证假设，不能将待跑实验写成结论。
- GiGPO 见 `docs/RELATED_WORK_ZH.md`；六组完整历史训练轨迹已做无卡状态重用审计，产物在
  `state_group_audit/`。它说明存在自然分组机会，不是GiGPO训练结果，也不证明probe更好。

## 交付前尚未完成的门槛

1. 提交并在干净 checkout 验证本轮新入口；随后按实际服务器部署。
2. 用获得授权的 GPU 验证 profile、显存、真实保存/恢复、SwanLab；24GB 候选可能仍需修正执行配置。
3. 同一硬件配置完成正式 12 组 × 200 steps，并产出全套 receipt、曲线、配对结果与失败/成本报告。
4. 按新结果更新技术复盘和简历表述，保留负面结果及协议限制，不反向改写旧实验。
