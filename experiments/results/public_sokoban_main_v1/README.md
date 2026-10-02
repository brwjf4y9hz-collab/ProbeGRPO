# 公开 Sokoban 主实验 v1 结果

本目录保存首个完整 ProbeGRPO 对比实验的精简版、已提交记录。原始 episode sidecar、trainer 日志和 checkpoint 位于 Git 忽略的 AutoDL `outputs/` 目录；每 seed CSV 根据其生成的 `summary.json` 转录，没有挑选有利运行。奖励为二元值，测试集有 128 项，因此 `success_count` 是各记录成功率所对应的精确分子。

## 固定实验协议

- 模型：`Qwen/Qwen3.5-2B`，LoRA rank 32。
- Trainer：当前 verl，commit `cf14ded3a448107e70a206fd201817cc1cbae348`。
- 环境来源：`ZihanWang314/ragen-datasets`，commit `e2060cf7c51da891a68e0e7437309e5ec052e45b`。
- 数据划分：512 个唯一训练棋盘和 128 个唯一测试棋盘，布局无重叠。
- 精确 BFS oracle：每个选中棋盘都可在 12 步 horizon 内解出。
- 训练：50 个 update，每个 update 4 个 prompt，每个 prompt 4 条主 rollout。
- 种子：17、42、101。
- 硬件：单张 48 GB RTX 4090。
- Probe 方法：budget 2，局部 credit 系数 0.5。

无需 torch、verl 或绘图库即可从已提交 CSV 重新生成结果：

```bash
make results
```

如需从 AutoDL 原始 summary 重建 CSV：

```bash
python3 scripts/aggregate_sokoban_seeds.py \
  --output-dir experiments/results/public_sokoban_main_v1 \
  outputs/ablation-bbba7a8-mem25-seed17 \
  outputs/ablation-main-v1-seed42 \
  outputs/ablation-main-v1-seed101
```

## 结果

生成的表格见 [`summary.md`](summary.md)，机器可读统计见 [`aggregate.json`](aggregate.json)。`+/-` 表示三个 seed 的样本标准差，不是置信区间。

当前最有依据的比较是 LinearUCB-B2 对 GRPO：

- 成功率：`31.0% ± 1.2%` 对 `24.0% ± 2.0%`；
- 配对提升：`+7.0 ± 0.8` 个百分点；
- 成功测试 episode 数：LinearUCB 在配对 seed 上分别多 9、10、8 个（每组 128 项）；
- 额外 rollout token：平均 `51.6%`，即主 rollout token 的 `1.516` 倍；
- 有效 probe：`1,138 / 1,200`（`94.8%`）。

Surprisal-B2 的数值均值最高（`31.2%`），但额外 rollout token 更多（`57.3%`），跨种子波动也更高（3.4 个百分点）。与 LinearUCB 相差 0.2 个百分点，远小于观测到的 seed 波动，不视为胜出。

## 证据边界

- 三个 seed 支持作品集级重复性记录，不支持正式显著性结论。
- LinearUCB 是本次跨种子波动最低的调度器，但**没有**赢得每千 token 高影响 anchor 指标。Random-B2 平均 `11.93`，LinearUCB `11.32`，Surprisal `10.44`，因此原先的 anchor 效率主张不成立。
- LinearUCB 未达到原定 `<=1.5x` rollout-token 成本目标，高出 1.7 个百分点。应报告实测 `1.516x`，不要向下取整掩盖。
- 当前 budget 2 是在每组四条 rollout 中先选前两条 episode，再各选一个 turn；不是全组 top-B 选择。
- Seed 17 的 GRPO 臂在 PPO mini-batch padding 修复前完成并被保留，因为其合成行的 loss mask 为零。Seed 42 和 101 全程使用最终配置。论文级主张前应重跑 seed 17 GRPO。

## 实验主机原始结果目录

```text
/root/autodl-tmp/ProbeGRPO/outputs/ablation-bbba7a8-mem25-seed17
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed42
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed101
```

失败的 padding 运行以 `random_b2.failed-padding` 保留；汇总器因该运行没有达到声明的最终 step 而将其排除。
