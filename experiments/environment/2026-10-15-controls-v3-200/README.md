# 200-step 代码准备验收（2026-10-15）

用户批准四组正式对照统一 200 updates；当前无卡，**没有启动新训练，也没有产生新性能结果**。

## 固定内容

- 新协议：`experiments/protocols/public_sokoban_controls_v3_200.json`。
- 四组 × seed17/42/101 × 200 updates，共 12 runs、2,400 个更新。
- 原模型 revision、verl commit、512/128 数据、LR、四个实验臂和 warm-up 参数与 v2 完全一致，
  有自动测试逐字段核对；不因无卡改小正式训练。
- 每 10 步验证，包含 step0 共 21 次/组；主终点 step200，step50/100/150 为预先指定诊断。
- step50/100/150/200 保存 LoRA-only checkpoint，原生保留最近两份 actor checkpoint，
  优化器/额外状态仍按原配置保存。实际 GPU 保存/恢复尚未执行。
- 独立输出 `outputs/final-controls-v3-200`、SwanLab 前缀 `controls-v3-200`；旧 v1 结果、
  v2 50-step 协议与旧验收记录不改。旧 150-step extended 入口移除，避免混淆。

## 已验证

运行 `make check`：94 tests，92 项通过，2 个 PyTorch 张量测试因本地没有 torch 跳过；
CPU data-flow smoke、AgentLoop smoke 和所有 shell 语法检查通过。原始日志见 `local-check.log`。

关键回归覆盖：

- 12 组计划与实际传给训练脚本的环境变量均为 200 steps；旧 shell 中的 1-step、小数据环境值不能覆盖。
- smoke 独立目录并在 step1 保存，不作为正式实验，也不改变正式 12 组规格。
- 200-step 完成验收逐步核对：结构测试包含 201 个 step、21 次验证和最终模型文件；
  有最终文件但训练日志止于 step50 时，正确在 update51 拒绝。
- 图表导出覆盖 201 个逐步点，SVG 横轴到 step200。
- 聚合拒绝 50-step manifest，且历史必须覆盖完整 0–200；不因只找到最终行就认定完成。
- v2 显式旧协议仍打印 12×50 的计划，默认新版为 12×200，两者路径无交叉。

测试中的小型 JSON/伪 checkpoint 只是结构单元测试样例，不是缩小后的实验或真实模型保存。

## 远端状态

原 SSH 地址两次返回 `Connection closed`。因此本轮**未同步、未在远端重新测试 200-step 版**。
上一条 v2 验收记录的“90 项全通过”只对当时的远端版本成立，不能套用到本轮。
恢复服务器连接后，先核对/备份/同步，再跑固定运行环境中的完整测试和 preflight。
有卡时仍需原规格 48GB GPU 完成 smoke 和正式矩阵；不得缩减数据、种子或更新数。

没有提交或推送 Git，没有创建新的 SwanLab 实验，没有改写旧结果。
