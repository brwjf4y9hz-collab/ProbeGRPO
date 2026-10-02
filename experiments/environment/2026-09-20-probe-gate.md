# Sokoban probe trainer gate — 操作员提供的控制台证据

来源：操作员于 2026-09-20 粘贴的 AutoDL 终端输出。原始日志和 episode sidecar 保存在 `/root/autodl-tmp/ProbeGRPO/outputs/`。此处为日志整理，不是独立重跑实验。

## ProbeGRPO `076136b` 的单步结果

- 首次启动在任何 rollout 之前就因 vLLM 初始化失败，原因是 `OMP_NUM_THREADS` 解析成非正值。通过 shell 和 Ray runtime environment 显式设为 `OMP_NUM_THREADS=1` 后，完成一次 actor update。
- 输出目录：`outputs/verl-sokoban-probe-train-076136b-omp1/`。
- `training/global_step=1`、`probe/valid=2`、`probe/skipped=6`、`probe/extra_tokens=875`、`probe/changed_tokens=4`、`probe/credit_abs_sum=2.0`。
- 八条保存的 episode 与八行 rollout 匹配。Sidecar 检查器确认 `budget=0` 和 `lambda=0` 时 GRPO advantage 不变。
- 两个有效 probe 的原始 delta 分别为 `1.0` 和 `0.0`。当时使用的中心化 z-score 将其映射为 `+1` 和 `-1`，每条 episode 各修改两个 assistant token。零 delta 上出现的负数来自 batch 内相对中心化，并不能说明事实动作更差。

## 解释与下一道 gate

这证明真实模型生成的成对 probe 进入了 verl v1 advantage hook 和 actor update。这**不是**性能对比或消融。此后归一化已改为保零的 `max(1, 最大绝对 delta)` 缩放，`[1.0, 0.0]` 现在映射为 `[1.0, 0.0]`。该改动仍需重新通过 sidecar 和 GPU 核验，之后才能开始同预算实验。不要将首次运行指标与后续实验混合。
