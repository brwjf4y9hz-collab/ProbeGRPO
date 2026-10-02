# 公开 Sokoban 消融运行记录 — 2026-09-21

## 复现范围

- ProbeGRPO 分支：`codex/verl-gpu-bootstrap`
- 模型：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`
- verl revision：`cf14ded3a448107e70a206fd201817cc1cbae348`
- 公开数据：`ZihanWang314/ragen-datasets`，revision `e2060cf7c51da891a68e0e7437309e5ec052e45b`
- 派生数据划分：512 个唯一训练棋盘、128 个唯一测试棋盘，已排除测试/训练布局重复。
- Horizon：12 步。精确 BFS 核验所有选中棋盘都能在 horizon 内解出；训练集最长最短解为 11 步，测试集为 9 步。
- 主 rollout 形状：每个 update 4 个 prompt，每个 prompt 采样 4 条，共 16 条轨迹。

源 parquet 和派生 parquet 的校验和写在生成数据的 `manifest.json` 中。每个训练臂都会在被忽略的 `outputs/` 目录写入 `run_manifest.json`、解析后的 trainer 日志、标准 rollout JSONL，以及每条轨迹一个完整 episode sidecar。原始输出可能很大，因此不会提交到 Git；结果汇总和故障记录会提交。

## 训练前评估

修正显存问题后的运行完成全部初始化，并评估公开测试集 128 个样本：

- 初始成功率/reward 均值：`0.2578125`（33/128）；
- verl 的 assistant/user turn 计数方式下，validation episode turns 最小值为 `2`、最大值为 `24`、均值为 `18.515625`。

这是优化器更新前的预训练模型基线，不是 ProbeGRPO 结果。

## 事件 1 — 公开数据无法下载

**现象：**出现 `Network is unreachable`，随后 Hugging Face HTTP client 关闭。

**原因：**AutoDL 无卡模式没有可用的网络加速路由。

**解决方式：**加载 `/etc/network_turbo`，仅在准备数据期间关闭离线模式，禁用 Xet，并把下载 timeout 设为 600 秒。数据生成前会校验不可变 SHA-256。

## 事件 2 — actor entropy CUDA OOM

**现象：**`entropy_from_logits` 尝试再申请 11.97 GiB 显存，但当时只剩 7.17 GiB。

**原因：**一次性对过多 token 物化全词表 entropy。

**修复（commit `f9f27bd`）：**启用 verl 分块 entropy 计算，chunk size 256；actor dynamic token budget 设为 2048；response 上限 1024；减小 PPO mini-batch。所有实验臂均使用修正后的同一配置。

## 事件 3 — 同机初始化 worker 被杀

**现象：**初始 vLLM 权重同步期间 FSDP actor worker 退出，没有 Python 异常；Ray 报 unexpected EOF。

**原因：**同机 vLLM server 预留了 48 GB GPU 的 45%，FSDP 同步权重时剩余瞬时显存不足。

**修复（commit `bbba7a8`）：**将 vLLM `gpu_memory_utilization` 降到 `0.25`。下一次运行完成 actor/vLLM 初始化和初始验证。

## 事件 4 — 合成 padding 没有 episode sidecar

**现象：**Random-B2 将 16 条真实轨迹补到 32 行，随后因合成行没有 probe sidecar 而触发 `FileNotFoundError`。

**原因：**`ppo_mini_batch_size=8` 使 v1 trainer 要求 32 行倍数。合成 padding 行不对应真实环境 episode，因此本来就不应有 replay trace。

**修复（commit `31e0da9`）：**将 `ppo_mini_batch_size` 设为 4，使所需倍数为真实轨迹数 16。这样避免合成训练行，而不是伪造 probe credit。消融启动器还会跳过已有最终 step sidecar 的实验臂，因此 GRPO 可保留，失败的 probe 臂可重跑。

## 证据边界

- Random-B2 首次 sidecar 故障前，GRPO 臂已完成；具体最终指标应读取已保存日志和生成汇总，不在此推算。
- 失败的 Random-B2 目录以 `.failed-padding` 后缀保留。
- 四个同预算实验臂完成且审阅生成的 `summary.json`/`summary.md` 前，不作提升结论。

## Seed 17 主消融结果

四个完整实验臂均达到 50 个 update。不完整的 `.failed-padding` 运行已从修正后的汇总中排除。

| 方法 | 最终验证成功率 | Probe token | Probe/主 rollout 开销 | 有效 probe | GPU 小时 |
|---|---:|---:|---:|---:|---:|
| GRPO | 0.242 | 0 | 0.000 | 0/0 | 0.255 |
| Random-B2 | 0.273 | 7,211 | 0.481 | 373/400 | 0.279 |
| Surprisal-B2 | 0.273 | 8,602 | 0.576 | 376/400 | 0.287 |
| LinearUCB-B2 | 0.312 | 7,814 | 0.519 | 376/400 | 0.283 |

在这个单 seed 上，LinearUCB-B2 比 GRPO 高 7.0 个百分点，比同预算 Random-B2 高 3.9 个百分点。这是有希望的工程证据，不是统计结论。每 1,000 个 probe token 找到的高影响 anchor 数量上，Random-B2（`12.620`）高于 LinearUCB（`10.878`），因此当前调度器尚未证明其 anchor 效率假设。发布简历提升结论前，应冻结配置并在 seed 42、101 重复。

## 三种子完成 — 2026-09-22

Seed 42 和 101 使用固定公开数据划分完成四个实验臂。生成的汇总如下：

| 方法 | 最终成功率（均值 ± 样本 SD） | 相对 GRPO 的配对提升 | Probe/主 rollout 开销 |
|---|---:|---:|---:|
| GRPO | 24.0% ± 2.0% | - | 0.0% |
| Random-B2 | 29.2% ± 4.6% | +5.2 ± 6.5 个百分点 | 51.2% |
| Surprisal-B2 | 31.2% ± 3.4% | +7.3 ± 4.3 个百分点 | 57.3% |
| LinearUCB-B2 | 31.0% ± 1.2% | +7.0 ± 0.8 个百分点 | 51.7% |

LinearUCB 在三个配对 seed 上都高于 GRPO，是波动最低的 probe 方法；均值上没有明显胜出，Surprisal 高 0.2 个百分点但成本更高。Random 仍在每千 token 的高影响 anchor 指标上领先，因此不声称 LinearUCB 的调度效率更高。这里只作描述性报告，不作显著性检验。

保留的 seed 17 GRPO 早于 PPO mini-batch padding 修正。合成 padding 行的 loss mask 虽为零，但记录中保留了配置差异作为限制。若要作论文级结论，应补跑一次 GRPO 消除该差异。
