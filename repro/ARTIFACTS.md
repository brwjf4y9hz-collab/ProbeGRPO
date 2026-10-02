# 归档产物目录 — 2026-10-02 保存记录

## 公开发布的、带版本标识的证据

发布标签：[`repro-2026-10-02-v1`](https://github.com/brwjf4y9hz-collab/ProbeGRPO/releases/tag/repro-2026-10-02-v1)。该发布页提供下列分别计算校验和的文件。具体大小和 SHA-256 值见 `release-assets.json`；使用前请先核对 `SHA256SUMS`。

| 文件 | 内容 | 用法 |
|---|---|---|
| `experiment-evidence.tar.gz` | 原始日志、rollout/probe sidecar 和运行清单；不含 checkpoint 与 Ray 会话文件 | 解压到产物根目录后运行 `scripts/reproduce_results.py` |
| `evidence-manifest.json` | 20,399 个证据文件各自的 SHA-256 | 运行 `scripts/artifact_manifest.py verify-archive` |
| `verl-upstream-cf14ded.tar.gz` | 固定提交对应的上游源码归档，含原始许可证和依赖锁文件，不含 `.git` | 离线源码参考/备份；常规安装器仍使用固定 Git 提交 |
| `SHA256SUMS` | 发布文件校验和 | Linux 使用 `sha256sum -c SHA256SUMS`；macOS 使用 `shasum -a 256 -c SHA256SUMS` |

发布版源码包含精确的 ProbeGRPO 补丁、不可变的模型/数据引用、观测到的运行环境版本和复现说明。源码归档不等于预装好的运行环境。重新分发框架、模型或数据时，须保留各自上游许可证。

## Git 之外的所有者备份

- AutoDL 上的原始项目和输出仍保存在原有持久化目录。
- 本地 `ProbeGRPO-archive-20261002/` 保存证据、来源信息和数据归档、完整输出文件清单及 checkpoint 备份；已按 20,412 条清单记录逐项校验。是否完成以 `BACKUP_STATUS.json` 为准；后缀为 `.part` 的文件代表未完成。
- `provenance.tar.gz` 保存原项目 Git bundle、未修改的上游框架归档、完整本地框架差异，以及实际训练包的环境记录。
- `data-snapshot.tar.gz` 保存九个原始/预处理输入文件及其清单；公开复现流程会下载固定版本的原始来源并重新生成数据切分。项目许可证不自动授予第三方数据集的新许可。
- 经独立校验的 Qwen 基础模型快照保存在所有者较早建立的 `ProbeGRPO-data-backup` 目录；`model-manifest.json` 列出了预期的 13 个文件。

完整输出的大型备份不属于 Git 对象，也不作为公开模型发布。它包含早期 GSM8K 第 6 步的 FSDP checkpoint 和优化器状态。主要 Sokoban 运行关闭了 checkpoint 保存，因此没有最终权重。

## 恢复顺序

1. 检出已保存的代码标签，并运行 CPU 检查。
2. 按 `REPRODUCE.md` 重建固定版本的框架和环境。
3. 下载并校验固定版本的模型和数据，或恢复所有者的输入备份。
4. 下载体积较小的证据归档，检查并重新计算历史结果。
5. 只有调查原始日志或 checkpoint 时才恢复完整输出，并按 `outputs-manifest.json` 校验。旧 Ray `session_latest` 符号链接记录了已经失效的绝对路径；复现结果不需要它。

至少有一份独立的完整备份通过清单校验之前，不要删除服务器上的原始数据。备份应与可随时重建的运行环境分开放置。
