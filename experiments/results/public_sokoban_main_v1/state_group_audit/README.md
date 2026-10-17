# v1 状态重用审计：GiGPO 对照是否有信号机会

2026-10-16 在原服务器无卡模式读取完整 sidecars；没有加载模型或重新训练。
范围是 GRPO 与 LinearUCB-B2、seed17/42/101，共六组 × 50 updates、4800 条训练 episode，
不是全部四方法十二组，也不是修复后的 v3。每个 update 核对四个原始 prompt UID、每个 UID 四条 session。
验证集完全排除；不合并不同 update、run 或重复抽到相同棋盘的不同 UID。

## 观测

下表分母是各 run 的训练动作 occurrence 数，表示其中有多少属于跨轨迹重复组。
它不是“有多少独立状态”、不是成功率，也不把这些动作当作独立统计重复。

| 方法 | seed | 动作数 | 严格 state hash | 棋盘 observation | 棋盘+剩余步数 | 记录文本历史 |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 17 | 7652 | 33.70% | 68.20% | 37.74% | 22.79% |
| UCB | 17 | 7413 | 33.98% | 70.39% | 37.43% | 25.23% |
| GRPO | 42 | 7422 | 35.83% | 71.81% | 40.02% | 25.24% |
| UCB | 42 | 7266 | 36.84% | 70.20% | 39.71% | 26.88% |
| GRPO | 101 | 7730 | 30.70% | 66.61% | 34.53% | 23.18% |
| UCB | 101 | 7686 | 35.75% | 69.62% | 39.06% | 25.24% |

所有 K 条轨迹本来就共享初始状态，所以还要看排除 turn0 的比例：棋盘分组的非初始跨轨迹覆盖
为 62.76%–68.41%，棋盘+剩余步数为 26.97%–32.77%，文本历史为 13.78%–17.83%。
因此不能以“状态几乎不重复”作为不补 GiGPO 的理由。

“重复”也不等于“有可用差异”。以棋盘分组，属于跨轨迹且动作、return 均有差异组的动作比例
为 15.19%–19.23%；改为棋盘+剩余步数时为 6.30%–7.22%。这些仍只是信号机会，
既没有测量 advantage 的质量，也没有证明动作的因果贡献，更不能据此预测 GiGPO 成功率。

## 四种键的含义

- full_state：原 replay hash，包含 steps、invalid_actions、seed 等。
- observation：仅棋盘；可能合并不同剩余时域和策略上下文。
- horizon_observation：棋盘与 `max_horizon - turn_index`；不声称包含所有环境字段。
- history：已记录 observation / assistant raw_text 序列；不是 token 级完全相同输入的验证。

所有 return 均为 gamma=1 的逐步 reward-to-go，已核对与二值终局 reward 一致。
这是离线诊断，不是 GiGPO 算法适配；没有修改 reward、advantage 或原始轨迹。

## 来源与复核

每个 run 子目录包含 `groups.csv` 和 `summary.json`；后者保存完整来源相对路径列表、
逐文件 SHA-256、manifest SHA-256 和来源组合指纹。大体积原始 episode 留在服务器 ignored outputs。
审计脚本为 `scripts/audit_state_groups.py`，仅依赖标准库。

例如，在原服务器仓库根目录运行：

```bash
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/audit_state_groups.py \
  outputs/ablation-main-v1-seed42/grpo --rollouts 4 --prompts 4 \
  --output-dir outputs/state-groups-grpo42-new
```

本轮实际脚本上传并运行于独立目录 `outputs/portfolio-prep-20261016-v1/`，未覆盖原训练代码。
六组目录分别是 `ablation-bbba7a8-mem25-seed17`、`ablation-main-v1-seed42`、
`ablation-main-v1-seed101` 下的 `grpo` 与 `linear_ucb_b2`。
seed17 GRPO 的旧配置差异仍存在；这个审计不能修复实验配置，也不是新的泛化结论。
