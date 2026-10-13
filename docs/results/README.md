# ProbeGRPO 结果图与分析

2026-10-13 整理已有实验；本轮没有新增训练或推理。

## 成功率、种子波动与探测成本


![ProbeGRPO 结果与成本](figures/01_probegrpo.png)

[可编辑矢量图 SVG](figures/01_probegrpo.svg) · [逐种子数据 CSV](../../experiments/results/public_sokoban_main_v1/per_seed.csv) · [聚合数据 JSON](../../experiments/results/public_sokoban_main_v1/aggregate.json)

模型 Qwen/Qwen3.5-2B，512 个训练关卡、128 个共同留出关卡；四方法各运行三个种子（17、42、101），每次 50 次训练更新。12 次训练运行对同一批留出关卡重复评估，不能写成 1536 道独立题。

| 方法 | 成功率均值 ± 样本 SD | 相对 GRPO 配对提高（pp） | 额外 rollout token | 有效探测 / 尝试 | 高影响 anchor / 千额外 token | 平均 GPU 时长（小时） |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 23.96% ± 1.97 pp | +0.00 | 0.00% | — | — | 0.231 |
| Random-B2 | 29.17% ± 4.58 pp | +5.21 | 51.21% | 1140/1200 | 11.93 | 0.280 |
| Surprisal-B2 | 31.25% ± 3.41 pp | +7.29 | 57.28% | 1142/1200 | 10.44 | 0.286 |
| LinearUCB-B2 | 30.99% ± 1.19 pp | +7.03 | 51.64% | 1138/1200 | 11.32 | 0.284 |

| 种子 | GRPO 成功 / 128 | LinearUCB-B2 成功 / 128 | 配对提高（pp） |
|---|---:|---:|---:|
| 17 | 31/128 | 40/128 | +7.03 |
| 42 | 28/128 | 38/128 | +7.81 |
| 101 | 33/128 | 41/128 | +6.25 |

图中误差线是跨种子样本标准差，不是置信区间；三个黑点是三次原始运行。LinearUCB 波动较小属于描述性观察，三种子不足以证明显著性或普遍稳定性。

额外 token 比例以各方法自己的主 rollout token 为分母；不是相对于 GRPO 的等总 token、等费用或等时长比较。LinearUCB 总 rollout token 比例为 1.516×，略高于原 1.5× 目标。Surprisal 的成功率均值更高；Random 的 anchor/千额外 token 诊断更高，因此不能写“LinearUCB 全面最优”。

历史限制：seed17 的 GRPO 在 padding 修复前运行；seed42、101 使用修复后的配置。主实验最终权重未保存，已有原始轨迹与评估数据。Anchor 分数是记录的局部反事实诊断，不能直接称为完整因果贡献。


完整历史协议见[结果记录](../../experiments/results/public_sokoban_main_v1/README.md)与[归档核验](../../repro/validation/README.md)。独立运行与评测工程项目见 [SWE-HarnessLab](https://github.com/brwjf4y9hz-collab/SWE-HarnessLab)；两个项目任务与指标不同，不合并计算成绩。

## 来源核验与离线绘图

图片为 PNG 与 SVG，数据来自仓库内既有报告。来源路径、SHA256、图文件哈希见 [figure-provenance.json](figure-provenance.json)。本次已检查中文字体、图例、注释与固定分母。

在项目根目录使用独立绘图环境：

```sh
python3.11 -m venv .venv-viz
.venv-viz/bin/python -m pip install -r docs/results/requirements-viz.txt
.venv-viz/bin/python docs/results/plot_results.py --output /tmp/project-result-figures
```

输出目录须不存在；需安装中文字体（如 Noto Sans CJK SC、Arial Unicode MS 或黑体）。绘图环境独立于项目核心依赖，命令只读取已封存数据，不训练、不请求模型、不启动官方评分。系统字体与版本差异可能改变排版；SVG 时间戳也会改变文件哈希，复现时先核对数据与数值口径。
