# ProbeGRPO：从一条有效训练信号，到可解释的三种子结果

[项目仓库](https://github.com/brwjf4y9hz-collab/ProbeGRPO) · [问题索引](README.md) · [结果图与分析](../results/README.md)

ProbeGRPO把反事实turn探测加入多轮GRPO。真正困难的部分不只是选一个anchor，而是让重放状态、奖励差、token mask、真实episode与actor batch保持一致，再诚实解释成功率和探测成本。这个复盘按训练gate、credit语义、显存与batch、数据和结果保存的顺序，记录实际遇到的故障与工程取舍。

早期gate先证明加载、更新和恢复，之后才接入真实Sokoban probe；公开数据与三种子实验在这些工程问题之后开展。早期单步日志来自当时操作员提供的控制台证据，本轮只复核仓库记录与当前实现，未访问GPU重跑。

## 先把不同层的失败分清

以下各项同时保留实际故障和主动设计约束。标题后的状态说明处理到什么程度；“通过”仅指列出的验收，不扩大成训练泛化或解题优势。先解释观察与定位，再讲采取的行动和验证结果。

<a id="p01"></a>

## 1. 训练完成并不意味着训练信号有效（P01）

**范围与状态：**训练栈与 gate；配置规避；GPU 冷启动复现待做。

首次五步 GRPO 的最后一批 reward/advantage 为零，响应均达到 256 token 上限；退出时还有 DataLoader worker 清理错误。 日志显示 8 条真实轨迹补成 32 行；该 verl revision 会把 actor mini-batch 乘以 rollout.n。短响应上限与合成行让“跑完几步”不足以证明真实任务信号有效。DataLoader 子进程被终止的根因没有完整证明。

**处理过程。**先用 GSM8K 验证加载、更新与恢复；单独做 signal smoke，将 mini-batch 设为 2、DataLoader workers 设为 0、响应上限调到 1024、token budget 设为 2048，检查真实轨迹、advantage 与梯度，而不是只看退出码。固定 verl 提交与 NumPy 2.3.5 覆盖并记录依赖检查。

**验收与结果。**修正后的单步保存恰好 8 行，reward 均值 0.75，梯度范数 0.800695，正常退出；另一次从 step5 恢复到 step6 成功，但设置不同，不能合并成同一 gate。

**边界。**单步 smoke 不是 ProbeGRPO 性能结果；actor 分配显存不是整机峰值。将 workers 设为 0 是已验证的规避，不能声称找到了 worker 退出的完整根因。

来源：[experiments/environment/2026-09-19-gpu-gate.md](../../experiments/environment/2026-09-19-gpu-gate.md)。

当前实现入口：[scripts/run_verl_grpo_smoke.sh](../../scripts/run_verl_grpo_smoke.sh)、[scripts/bootstrap_verl.sh](../../scripts/bootstrap_verl.sh)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p02"></a>

## 2. OMP 线程设置被解析为非正值（P02）

**范围与状态：**环境配置；记录配置修正后的单步通过。

Sokoban probe 首次启动在任何 rollout 之前因 vLLM 初始化失败；OMP_NUM_THREADS 被解析为非正值。 这是初始化环境问题，不是 rollout、probe 调度或 actor 更新失败。历史证据来自操作员保存的控制台输出。

**处理过程。**在启动 shell 与 Ray runtime environment 中都显式设置 OMP_NUM_THREADS=1，再执行同一类单步 gate；保留失败记录。

**验收与结果。**后续完成一个 actor update，记录 2 个有效 probe、875 额外 token、4 个 changed tokens，并保存 8 条匹配的 episode/rollout。

**边界。**这只证明成对 probe 接入了更新链路，不证明任务成功率提高；首次 gate 使用的归一化后来仍需要纠正。

来源：[experiments/environment/2026-09-20-probe-gate.md](../../experiments/environment/2026-09-20-probe-gate.md)。

当前实现入口：[scripts/run_verl_sokoban_smoke.sh](../../scripts/run_verl_sokoban_smoke.sh)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p03"></a>

## 3. 零 delta 被中心化成负 credit（P03）

**范围与状态：**算法语义错误；保零规则已实现；历史输出保留。

最早两个有效 probe 的原始 delta 为 [1.0, 0.0]，中心化 z-score 却得到 [+1, -1]，零差值也改变 assistant turn 的 advantage。 相对中心化把“没有奖励差”解释成“低于这批 probe 均值”；这与零 delta 不应提供额外局部 credit 的语义冲突。

**处理过程。**改为除以 max(1, 最大绝对有效 delta)，不减均值，只将 credit 加到对应 assistant turn mask；budget=0、lambda=0 与 zero delta 分别保留基线行为检查。

**验收与结果。**当前实现将 [1,0] 映射为 [1,0]，零 delta 仍为零；相应代码与后续主实验结果封存，首次 z-score gate 不作为最终性能对照。

**边界。**局部单替代动作的奖励差仍不是完整因果贡献；通过数值不变式也不能单独证明最终任务收益。

来源：[experiments/environment/2026-09-20-probe-gate.md](../../experiments/environment/2026-09-20-probe-gate.md)、[experiments/results/public_sokoban_main_v1/README.md](../../experiments/results/public_sokoban_main_v1/README.md)。

当前实现入口：[src/probegrpo/advantage.py](../../src/probegrpo/advantage.py)、[tests/test_pipeline.py](../../tests/test_pipeline.py)、[tests/test_verl_v1_sidecar.py](../../tests/test_verl_v1_sidecar.py)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p04"></a>

## 4. 同机推理与训练争抢显存（P04）

**范围与状态：**资源与初始化；后续初始化和主实验完成；部分归因来自日志。

actor entropy 计算申请额外 11.97 GiB 时只剩 7.17 GiB；另一轮 FSDP worker 在初始 vLLM 权重同步时退出，Ray 报 EOF，没有 Python 异常。 entropy OOM 明确指向全词表熵物化。初始化 worker 被杀在记录中被归因于同机显存竞争；没有足够信息把所有无异常退出都证明为同一根因。

**处理过程。**启用分块 entropy，chunk size 256；限制 actor dynamic token budget 为 2048，响应上限 1024，调整 mini-batch。另将 vLLM gpu_memory_utilization 从 0.45 降到 0.25，为权重同步留空间。

**验收与结果。**配置修正后的运行完成初始化与 128 关卡初始验证；后续四方法三种子主实验完成。相关修正分别在历史 commit f9f27bd、bbba7a8 记录。

**边界。**成功重跑支持配置有效，不等于完整证明瞬时显存根因；没有跨 GPU/模型规模的吞吐或峰值显存研究。

来源：[experiments/environment/2026-09-21-public-ablation-incidents.md](../../experiments/environment/2026-09-21-public-ablation-incidents.md)。

当前实现入口：[scripts/run_public_sokoban_train.sh](../../scripts/run_public_sokoban_train.sh)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p05"></a>

## 5. 合成 padding 行没有真实 episode sidecar（P05）

**范围与状态：**训练数据与 replay；真实行形状修正；seed17 对照差异保留。

Random-B2 将 16 条真实轨迹补到 32 行后，在读取合成行 probe sidecar 时触发 FileNotFoundError。 ppo_mini_batch_size=8 结合 rollout.n 要求 32 行倍数；合成 padding 不是环境 episode，没有可以合法重放的前缀和 trace。

**处理过程。**mini-batch 改为 4，使要求倍数匹配 16 条真实 rollout；不伪造 sidecar 或 credit。失败目录保留 .failed-padding，汇总器拒绝没有达到声明最终 step 的运行。

**验收与结果。**修正后 probe 臂完成 50 updates；最终三种子统计明确列出 seed17 GRPO 早于 padding 修正的历史差异。

**边界。**seed17 GRPO 未以修正后配置补跑；零 loss mask 不足以消除协议差异。论文级因果对照仍需消除该差异。

来源：[experiments/environment/2026-09-21-public-ablation-incidents.md](../../experiments/environment/2026-09-21-public-ablation-incidents.md)、[experiments/results/public_sokoban_main_v1/README.md](../../experiments/results/public_sokoban_main_v1/README.md)。

当前实现入口：[scripts/run_public_sokoban_train.sh](../../scripts/run_public_sokoban_train.sh)、[scripts/summarize_sokoban_ablation.py](../../scripts/summarize_sokoban_ablation.py)、[tests/test_ablation_summary.py](../../tests/test_ablation_summary.py)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p06"></a>

## 6. 数据来源、重复布局和不可达关卡会影响结果解释（P06）

**范围与状态：**下载故障与实验设计；下载记录保留；去重与 oracle 校验完成。

公开数据下载曾报 Network is unreachable；即使下载成功，训练/测试布局重复和超过 horizon 的不可解关卡也会削弱评测含义。 历史日志把下载失败定位到 AutoDL 无卡模式网络路由；布局重叠与可达性属于主动检查的设计风险，不是已经观测到的泄漏事故。

**处理过程。**记录准备阶段网络调整与 timeout/Xet 设置；固定来源 revision 与 SHA。按棋盘布局去重、排除 train/test 重叠，用精确 BFS 检查 12 步 horizon 内的可达性。

**验收与结果。**得到 512 个唯一训练棋盘和 128 个唯一测试棋盘；记录全部可解，最长最短解为 train11/test9。后续归档核验中预处理 parquet 哈希与原记录匹配。

**边界。**这是同一 Sokoban 子集的评测，不代表跨环境泛化；12 次运行对共同128关卡的1536条记录不是1536道独立题。

来源：[experiments/environment/2026-09-21-public-ablation-incidents.md](../../experiments/environment/2026-09-21-public-ablation-incidents.md)、[repro/validation/README.md](../../repro/validation/README.md)。

当前实现入口：[scripts/prepare_public_sokoban_data.py](../../scripts/prepare_public_sokoban_data.py)、[tests/test_public_sokoban.py](../../tests/test_public_sokoban.py)。代码链接说明现行实现，不能替代当时版本日志。

<a id="p07"></a>

## 7. 成功率提高，但效率目标没有全面达到（P07）

**范围与状态：**负结果与实验解释；如实封存；未证明调度器最优。

LinearUCB 成功率高于 GRPO，但高影响 anchor/千额外 token 没有超过 Random；总 rollout token 比例 1.516×，略高于原 1.5× 目标。 任务成功率、跨种子波动、probe 有效性和 anchor 单位成本衡量不同问题；同 B=2 不等于等总 token 比较。

**处理过程。**同时报告四方法、三个种子、配对差值、样本 SD、token 与 GPU 时长；结果图保留 Random/Surprisal 的优势项，不只挑最好看的 LinearUCB 指标。

**验收与结果。**LinearUCB 30.99% 对 GRPO23.96%，配对均值+7.03pp，额外token51.64%；Random 的诊断11.93高于LinearUCB11.32，Surprisal的成功率均值31.25%更高。

**边界。**三种子不证明显著性；历史配置差异仍在。不能声称最优调度、普遍稳定、严格等成本收益或完整因果贡献。

来源：[experiments/results/public_sokoban_main_v1/README.md](../../experiments/results/public_sokoban_main_v1/README.md)、[experiments/results/public_sokoban_main_v1/aggregate.json](../../experiments/results/public_sokoban_main_v1/aggregate.json)、[docs/results/README.md](../../docs/results/README.md)。

<a id="p08"></a>

## 8. 归档能重算结果，却没有主实验最终模型（P08）

**范围与状态：**产物保存与复现；数据和日志核验完成；最终权重缺失未恢复。

主实验关闭 checkpoint 保存；现存大型 checkpoint 是早期 GSM8K step6。数据归档第一次整包核验因额外 metadata 文件拒绝，旧四舍五入 CSV 的 token 开销比例显示也偏粗。 产物是否存在、文件清单是否匹配与能否重训是三件事。数据拒绝来自归档布局，不是数据本身 SHA 不匹配；缺失最终权重不能从日志补造。

**处理过程。**按清单核验实验、模型和数据备份，单独核验解压数据目录；从原精度重建 CSV并保存legacy版。将代码、固定框架 patch、原始证据、模型/数据、checkpoint 分开归档。

**验收与结果。**历史核验记录：12次运行1536条评估、128共同棋盘、50步；20,399证据条目与20,412完整备份条目匹配；token 开销比例显示由51.7%修正为51.6%，成功次数不变。

**边界。**最终 Sokoban 权重仍不存在，GPU 冷启动重训未执行。公开档案复核结果不等于从头训练复现，历史服务器清单不等于本轮重新访问GPU。

来源：[repro/validation/README.md](../../repro/validation/README.md)、[repro/ARTIFACTS.md](../../repro/ARTIFACTS.md)。

当前实现入口：[scripts/reproduce_results.py](../../scripts/reproduce_results.py)、[scripts/artifact_manifest.py](../../scripts/artifact_manifest.py)。代码链接说明现行实现，不能替代当时版本日志。

## 收尾判断与未解决部分

主实验支持三种子描述性收益，但没有证明调度器最优或严格等成本优势。下一项研究验证应先处理seed17配置差异与最终权重保存，再预注册更多种子或环境；这些是未来建议，不是已经完成的动作。

主实验最终权重缺失，不能从现有日志恢复。仍未完成：全新GPU重训、worker退出完整根因证明，以及跨环境泛化与严格等总token对照。既有CPU检查和归档哈希不能替代这些验证。

本复盘由既有仓库记录与当前代码整理；没有补造个人独立贡献、操作细节、实验参数或因果结论。历史预算记录作为故障背景保留，不代表新的金额限制。
