# 验证记录 — 2026-10-02

| 检查项 | 结果 | 证据 |
|---|---|---|
| 全新 Python 3.9 虚拟环境中的可编辑安装 | 升级旧版自带 pip/build 工具后通过 | README 现已明确先升级 pip |
| `make check` | 共运行 78 项：76 项通过，2 项仅因缺少 torch 而跳过；两项 CPU smoke 和 shell 语法检查均通过 | `cpu-check.log` |
| 仅源码方式重建框架 | 重新获取精确提交并应用补丁成功；再次运行时确认补丁已存在 | `framework-setup.txt` |
| 对 trainer 修改的意外应用 | 已拒绝并保留拒绝记录 | `framework-refusal.json` |
| 已归档的原始实验记录 | 12 次运行、1,536 次最终评估、128 个相同且唯一的棋盘、最终更新步数 50；成功次数和成本差异均在 1e-12 以内 | `result-verification.json` |
| 证据归档 | 20,399 个条目全部与服务器文件清单一致 | `evidence-verification.json` |
| 含早期 checkpoint 的完整输出备份 | 20,412 个条目全部匹配；压缩后 5,015,429,285 字节 | 所有者备份：`full-backup-verification.json` |
| 模型备份 | 固定快照中的 13 个文件均通过 SHA-256 校验 | `model-verification.json` |
| 数据备份 | 九个文件均通过校验 | `data-backup-verification.json` |
| 在原训练环境中重新生成公开预处理数据 | 训练集和测试集 parquet 的哈希均与记录文件匹配 | `data-rebuild.json` |
| 全新 GPU 安装、训练和 checkpoint 恢复 | 未运行；未分配 GPU | `../../REPRODUCE.md` 中列出的待办项 |

第一次校验数据归档时，归档内额外包含的 `data-manifest.json` 元数据文件导致整体检查拒绝；随后按该清单独立校验了解压后的 `data/` 目录。此问题来自归档布局，不是数据校验和不匹配。

原始四舍五入后的 CSV 保存在 `experiments/results/public_sokoban_main_v1/legacy/`。改用原始精度后，LinearUCB 显示的成本从 51.7% 变为 51.6%；奖励次数和成功率不变。原始实验日志未改动。

CPU 检查不能证明 GPU 训练可复现。历史主实验关闭了 checkpoint 保存，因此没有主实验最终模型。早期 GRPO 第 6 步的 checkpoint 不应被描述为已训练完成的 Sokoban 结果。

提交 `1296593` 上的 GitHub Actions 已通过全部四项 push/PR 检查，覆盖 Python 3.9 和 3.12，并包含全新框架安装流程。
