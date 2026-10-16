# ProbeGRPO v3：200 步修复与补充对照

状态：2026-10-15 准备。**v1 的 12 组训练结果保留不变；以下是新代码与待运行协议，不是新性能结果。**

用户已批准把四组正式实验统一延长至 200 steps。新的固定协议为
`experiments/protocols/public_sokoban_controls_v3_200.json`；旧 v2 的 50-step 准备文件保留不改，
不混合终点、不复用旧输出目录。现在没有 GPU，默认命令只打印计划，不自动启动训练。
硬件选择见 [HARDWARE_PROFILES_ZH.md](HARDWARE_PROFILES_ZH.md)：学校单卡/双卡 24GB 为候选配置，
尚未实测能否运行；硬件配置不允许改数据、模型、全局 batch、K、B、λ 或 200-step 预算。

## 这轮改什么

1. 修正 LinearUCB warm-up 固定桶顺序：每轮随机打乱非空桶，再从各桶取一个 turn。
   真实 AgentLoop 是每个被选中的 episode 调用 `select(..., budget=1)`；全局 B2 是选两个
   episode，不是一次调用取两个 turn。固定桶顺序使单 anchor 优先落在最早的非空桶。
2. warm-up 分桶改用 `turn_id / max(1, trajectory_turns - 1)`，使提前结束的轨迹也能覆盖自身
   的前、中、后段。长度为 1 时只有前段，长度为 2 时没有中段；没有合法候选的桶不会被凭空补齐。
   为避免同时改变 UCB 的特征定义，`turn_fraction = turn_id / max_horizon` 保持为 horizon-relative
   特征，新增 `trajectory_fraction` 专用于分桶。旧记录没有实际长度时兼容旧坐标。
3. **不缩短 warm-up、不改学习率、不加熵奖励**。20 updates / 200 有效且有成本 probe 两个条件
   都满足后才进入排名。200-step 版本只延长训练，不同时调整 warm-up 和 credit 超参数。
4. 在 step50/100/150/200 保存 checkpoint，每个 run 最多保留最近两份 actor checkpoint；
   初始 smoke 为 step1 保存。启用固定 verl 的 `save_lora_only`，仍保存 optimizer/extra。
   它是 FSDP 原生 LoRA-only checkpoint，**不是已验证可分发的 PEFT adapter 导出**；新配置恢复训练
   仍须通过 GPU gate。旧主实验启动器关闭过 checkpoint，不能把早期 gate 的 checkpoint 当作主实验模型。
5. manifest 记录实际源文件 SHA-256、未提交状态、完整本层 Hydra 参数、安装的 trainer 哈希、
   包版本及数据文件校验。下层启动器的默认值可由源码指纹追踪，完整最终配置仍以训练日志为准。
6. 完成检查核对每一步训练/验证数量、有限数值、零 credit 对照和最终模型文件。
   只有最终 step 目录存在不足以认定完成；必须逐步覆盖 1–200，且有 step200 模型。
   中断目录保留，禁止自动覆盖；周期 checkpoint 不代表已经实现自动断点恢复，恢复仍需单独验证。

## 固定对照矩阵

所有组使用同一模型 revision、512/128 split、LR=3e-6、4 prompts/update、**200 updates**、
种子 17/42/101。完整矩阵共 12 runs，**不是已经跑完，也不是预计只需 1 GPU 小时**。
总计 2,400 个 optimizer updates。每组 step0 验证，随后每 10 步验证，总计 21 次验证/组，
每次 128 个 held-out 棋盘。正式主终点固定 step200；step50/100/150 只作预先指定的动态诊断。

| 组 | 主 rollout K | probe B | λ | 回答的问题 |
|---|---:|---:|---:|---|
| GRPO | 4 | 0 | 0 | 同一新版管线下的基线，包含重跑 seed17 |
| GRPO-K6 | 6 | 0 | 0 | 更多主采样是否能解释改进 |
| UCB probe-only | 4 | 2 | 0 | 做 probe 但不用于 advantage 时会怎样 |
| ProbeGRPO | 4 | 2 | 0.5 | 修复 warm-up 后的局部 credit 方法 |

Random-B2 / Surprisal-B2 的 v1 结果继续保留作既有调度器比较，不冒充新版同配置结果。
如需正式宣称新版调度器优越性，必须另跑同版 Random/Surprisal；这轮不自动扩张为全部消融。

K6 每步产生 24 条主轨迹，K4 产生 16 条；`ppo_mini_batch_size=4` 分别对应 24/16 的整除要求，
不需要旧 seed17 那样的合成 padding。K6 也改变了 GRPO 的组内估计，因此不是“只改算力”的完美干预。
λ=0 仍运行 probe，但额外调用可能改变随机状态/调度，不能承诺它与无 probe 基线参数逐位一致。
张量级的 λ=0 exact fallback 与整次随机训练完全相同是两回事。

### 成本口径与判定

- K6 只是**更多 rollout 对照**，不是事先认定的等 token、等 FLOPs、等 GPU 小时。
- 原始 `curves/metrics_by_step.csv` 保存主生成 token、probe token、累计 token 和累计 step 时间。
  先逐种子报告 `C = main_generated_tokens + probe_extra_tokens`、K6/ProbeGRPO 的 C 比及时间比。
- 预先约定：终点 C 比相差不超过 10% 时，只称“近似生成-token 匹配”；仍不是算力完全匹配。
  超出范围就明确标为未匹配，禁止为了好看的终点调数据。需要继续做严格 token-budget 实验时单列新协议。
- 累计 token 图排除 validation 和 prompt/prefill token；`timing_s/step` 之和是训练循环时间代理，
  不是账单 GPU 小时，也不是 profiler 测得的有效计算量。总进程耗时、初始化、验证与 checkpoint 耗时另看日志。
  checkpoint 时间仅在原日志提供该项时汇总，缺失时为 null，不补成“零成本”，也不与 step 总时间重复相加。
- 只比较预定最后一步，不用 held-out test 挑最优 checkpoint。三种子报告配对差值与样本 SD，
  同时报告每组 step0→终点的自身变化。不能由“基线下降、方法上升”直接断言终点选择有偏。
- 200×4=800 次训练 prompt 暴露，约 1.5625 个数据集遍历量；不保证收敛。
  四组都运行三个种子，不把单种子长跑或一步 smoke 当正式结果。原先可选的 150-step `extended`
  入口已移除，避免比新的正式训练还短却被误称为“扩展”。

## 无卡准备与有卡运行

以下命令从仓库根目录运行。无卡 preflight 不加载模型、不导入 torch/vLLM、不修改 verl、不创建训练实验。

```bash
VERL=/root/autodl-tmp/probegrpo-runtime/verl
PROTOCOL_FILE=experiments/protocols/public_sokoban_controls_v3_200.json
"$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" --preflight
```

48GB 基准机器先通过 GPU runbook 的主机检查；24GB 候选使用硬件 runbook 的对应检查，不能
直接把原单卡 45,000 MiB 门槛调低然后宣称适配完成。再明确启动一更新 smoke（四组、seed17）；检查 K6 是否无 padding、
λ=0 的 `probe/changed_tokens` / `probe/credit_abs_sum` 是否为零、checkpoint 是否真的可恢复。
原生保存逻辑已经核对，实际模型保存/恢复仍需要 GPU 验证。

```bash
SWANLAB_ENABLED=1 SWANLAB_MODE=cloud \
  "$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" --phase smoke --execute
```

确认 smoke 后，再显式运行完整 12 组（不默认后台开启或自动收费）：

```bash
SWANLAB_ENABLED=1 SWANLAB_MODE=cloud \
  "$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" --phase main --execute
```

可分批：`--phase confirm` 是 GRPO seed17 + 修复 UCB 三种子共四组；`--phase controls` 是剩余八组。
二者与 `main` 使用相同目录，完成且重新验收通过的组跳过；未完成目录不会被覆盖。
`--output-root` 可改到新的空目录；不要指向 v1 输出。默认只打印计划，只有 `--execute` 会启动模型。
默认输出为 `outputs/final-controls-v3-200/single_48gb/seed{17,42,101}/<method>/`；
`--output-root` 指定的是基础目录，启动器仍会追加硬件 profile 名。
SwanLab 名称带 `controls-v3-200`、硬件 profile 和 `steps200`，与旧 50-step 实验分开。
无卡测试不写入正式云端历史。同一源码/协议/硬件的四组 smoke 均通过后，主训练入口才会放行。
每次启动保存 `process_execution.json`（含失败退出码与 launcher 总耗时），完整 `main` 结束后自动
生成 `control_summary.json`：逐种子回报、自身增益、配对差值、实际 token 比、checkpoint 时间与时间比，
并单列预先指定的 step50/100/150/200 验证值。每组 `curves/training_curves.svg` 自动覆盖完整 0–200 横轴，
无需为更长训练手工截断曲线。
分批完成后可手动运行
`scripts/summarize_final_controls.py outputs/final-controls-v3-200/single_48gb`。
同时生成可读的 `control_summary.md`。汇总拒绝混合 profile/不同 GPU 型号组合、验收后改变的 CSV、
重复 step 和 NaN/Inf；双卡验收必须有两个 rank 的模型分片。
`completion_receipt.json` 单列 train、历次验证、最终验证的停止原因、非法动作类型和 probe 失败分类；
这些统计进入完整结果 JSON，不用“失败了”一句话掩盖原因，也不把历次验证合计误称为最终样本量。
缺少任意一组时只记录缺失列表，非零退出，不生成“完整主结果”的均值。
50-step manifest 或不足 201 行（含 step0）的历史不能作为 200-step 完整结果汇总。
命令显式传入新协议：若远端仍是旧启动器或新 JSON 尚未同步，应立即报错，而不是悄悄跑回 50 steps。

如仅需复查旧 v2 的 50-step 准备计划，显式使用
`--protocol-file experiments/protocols/public_sokoban_controls_v2.json`；它不会修改默认的 200-step 规格。
不要在新旧版本间续跑或拼接曲线。磁盘不足时停止并保留成果，扩容/另存后再处理，不能缩减种子或步数。

## 可核验审计

`scripts/audit_anchor_positions.py` 从原始 sidecar 生成每个 anchor 的来源路径、文件哈希、
实际/上限位置、可选桶集合和阶段统计。它不会把无候选跳过记录当成进入 UCB 排名。
v1 审计产物放在 `experiments/results/public_sokoban_main_v1/anchor_audit/`，不是口头发现。
不能仅从 `turn_id/max_horizon` 推导“所有候选必然落在前段”，必须检查实际可选位置。

熵变化只作诊断，不称为熵塌缩结论；三种子和非单调曲线不足以证明该机制。
等成本、λ=0、warm-up、种子/配置一致性是不同证据缺口；补一组对照不会自动证明完整因果链。
