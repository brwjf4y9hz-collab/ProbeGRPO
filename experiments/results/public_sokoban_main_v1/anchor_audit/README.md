# v1 anchor 位置审计

2026-10-15 从远端三组 UCB 原始 episode sidecars 重建；不是修复后训练结果。
`anchors.csv` 为逐 anchor 记录，带来源相对路径和原始文件 SHA-256；`summary.json` 含每种子统计
及全部 sidecar 内容/路径的组合指纹。只统计真的存在 `scheduler_turn` 的训练 anchor，不把跳过记录算作排名。

- warm-up 选中 612 个 anchor：horizon-relative 前/中/后段分别 610 / 2 / 0。
- 445 次 warm-up 实际同时有三个桶的合法候选；另有 36 次存在前、中两桶。
  因此偏差不只是提前结束或位置分母造成，固定桶顺序是直接原因。
- 按实际轨迹长度重算位置，旧 anchor 的前/中/后段分别 541 / 16 / 55。
  这也说明“环境前段”和“轨迹前段”不是同一个定义；不能混用坐标解释旧结果。
- 三个种子最后一次 warm-up anchor 均为 step27，首次真实排名 anchor 均为 step28。
  这是原始记录观察，不是简单用 200/8 推算。
- 新版为 warm-up 随机打乱非空桶，按实际轨迹长度分桶；UCB 的 horizon-relative 特征保留。
  修复不反向更改 v1 的选择、回报或成功率。

在服务器仓库根目录重跑（不加载模型）：

```bash
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/audit_anchor_positions.py \
  outputs/ablation-bbba7a8-mem25-seed17/linear_ucb_b2 \
  outputs/ablation-main-v1-seed42/linear_ucb_b2 \
  outputs/ablation-main-v1-seed101/linear_ucb_b2 \
  --output-dir outputs/anchor-audit-recheck
```

## 对外部评审说法的核对

数据源为本结果的 `training_history/per_step.csv`，不是另跑实验。
由原始逐步 `timing_s/step` 相加，UCB/GRPO 的时间代理增幅为：全三种子 22.986%，
仅配置更一致的 seed42/101 为 30.018%。`per_seed.csv` 的小时值经过四舍五入，
会得到略不同的百分数；不得称为已测得的账单 GPU 时间或精确 FLOPs。

UCB 相对 GRPO 的验证均值差值（pp）：

| step | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| 差值 | +0.260 | -0.521 | +0.781 | +3.646 | +3.906 | +7.031 |

所以 step20→50 的采样检查点差距递增成立，但“从头到尾单调扩大”不成立。
50-step 终点是预定协议，不能仅因曲线方向不同就认定人为挑点；也不能据此宣称已经收敛。
扩大训练、增加种子或等成本对照的结果仍待实际运行。
