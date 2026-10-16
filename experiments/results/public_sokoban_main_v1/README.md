# 公开 Sokoban 主实验 v1 结果

本目录保存首个完整ProbeGRPO对比的精简记录。原始episode sidecar和trainer日志位于AutoDL归档；主实验当时没有保存最终checkpoint，早期GPU gate权重不能冒充主实验权重。每seed CSV按归档精确记录恢复，没有挑选有利运行；128项二值评估的success_count是精确分子。

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
- LinearUCB未达到原定`<=1.5x`目标，精确均值约1.5164×，高约1.64个百分点；旧舍入记录会显示1.517×。
- 当前 budget 2 是在每组四条 rollout 中先选前两条 episode，再各选一个 turn；不是全组 top-B 选择。
- Seed 17 的 GRPO 臂在 PPO mini-batch padding 修复前完成并被保留，因为其合成行的 loss mask 为零。Seed 42 和 101 全程使用最终配置。论文级主张前应重跑 seed 17 GRPO。
- 旧warm-up存在最早非空桶偏置；修复后代码尚无新GPU结果。50步不是收敛证据，额外token与credit的作用没有被旧实验分离。

## 补充证据与后续协议

- [原始逐步曲线](training_history/README.md)：12runs、600updates、72次验证，无平滑，±为样本SD。
- [正/负/零credit案例](trajectory_demo/README.md)：来源哈希、单动作重放；不伪造未保存的完整suffix。
- [anchor审计](anchor_audit/README.md)：612个warm-up anchor及候选桶机会，首次真实排名step28。
- [状态重用审计](state_group_audit/README.md)：六组GRPO/UCB、4800条训练episode；不是GiGPO训练。
- [内存与时延](engineering_audit/summary.md)：旧12组log指标，不是匹配优化前后对照。
- [新200步协议](../../../docs/FINAL_CONTROLS_ZH.md)：独立目录与正式验收，不能与本目录混算。

## 实验主机原始结果目录

```text
/root/autodl-tmp/ProbeGRPO/outputs/ablation-bbba7a8-mem25-seed17
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed42
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed101
```

失败的 padding 运行以 `random_b2.failed-padding` 保留；汇总器因该运行没有达到声明的最终 step 而将其排除。
