# ProbeGRPO 复现指南

## 每条命令能验证什么

| 目标 | 输入 / 硬件 | 证据与限制 |
|---|---|---|
| CPU 测试与 smoke | Python >=3.9；无需模型/GPU | 检查核心重放、credit 和数据流；张量测试需要 torch |
| 已提交结果重绘 | Python 标准库 | CSV → summary/JSON/SVG；不会独立校验原始数据 |
| 重建原始结果 | 已保存的实验数据 | 12 次运行 × 每次 128 条最终评估；held-out 棋盘一致；50 个 update；含原始成本 |
| 重建框架 | Git/网络 | 固定上游 commit 和 trainer patch；不执行 GPU 训练 |
| 重建数据 | 训练环境 + 固定原始文件 | 重建 512/128 parquet 划分；输出哈希与记录文件一致 |
| GPU smoke/训练 | Linux NVIDIA CUDA、模型、数据、大容量持久磁盘 | 提供步骤；清理仓库期间未在全新 GPU 环境重跑 |

## 1. 干净 checkout 与最小 smoke

```bash
git clone https://github.com/brwjf4y9hz-collab/ProbeGRPO.git
cd ProbeGRPO
# 如需复现保留快照，请 checkout repro/ARTIFACTS.md 中列出的 release tag。
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
make check
make results
```

`make check` 会运行单元测试、CPU smoke、Agent 数据流 smoke 和 shell 语法检查。没有 torch 时，两个张量相关检查会跳过；这不能验证 GPU 路径。`make results` 使用全精度 CSV 重新生成已提交汇总和图。

## 2. 恢复并核验原始实验记录

从 [制品清单](repro/ARTIFACTS.md) 链接的 release 下载小型证据归档和 `evidence-manifest.json`，然后在仓库根目录运行：

```bash
python3 scripts/artifact_manifest.py verify-archive /path/to/experiment-evidence.tar.gz \
  --prefix outputs --manifest /path/to/evidence-manifest.json
mkdir -p artifacts/original
tar -xzf /path/to/experiment-evidence.tar.gz -C artifacts/original
python3 scripts/reproduce_results.py --artifact-root artifacts/original \
  --output-dir artifacts/recomputed
```

检查 `artifacts/recomputed/verification.json` 和 `summary.md`。脚本会校验全部 1,536 个最终二元奖励、相同的 128 个唯一测试棋盘、数据/运行元信息、step 50 和全精度成本；与仓库汇总不一致时会失败。原始证据不会被改写。结果匹配只能核验这些已归档观察，不能证明未来随机训练得到相同分数。

## 3. 框架与实际环境

上游固定版本：`repro/verl.commit`（`cf14ded3a448107e70a206fd201817cc1cbae348`）。代码改动：`patches/verl-probegrpo.patch`，共增加 16 行。`.gitignore` 改动和 trainer 备份文件不是复现 trainer 行为所必需的。

只重建源码（CPU 机器也可执行）：

```bash
bash scripts/setup_verl.sh /absolute/path/to/new-verl
# 再执行一次是安全的：脚本会检查精确 patch 是否已应用。
bash scripts/setup_verl.sh /absolute/path/to/new-verl
```

遇到意外 trainer 修改或上游 commit 不匹配时，脚本会失败并保留已有文件。当前 setup 脚本替换了不完整的历史脚本。

在 Linux NVIDIA 主机上使用一个**新的**持久化运行目录：

```bash
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
export VERL_DIR=/path/to/persistent-workspace/probegrpo-runtime/verl
```

环境使用 Python 3.12.3、uv 0.12.15、上游 `uv.lock` 的 `vllm` 和 `fsdp` extras，再覆盖为 NumPy 2.3.5。项目以 editable 模式安装但不重新解析依赖，然后执行 `uv pip check`。实际历史训练环境版本记录在 `repro/runtime-observed.txt` / `runtime-environment.json`：torch 2.11.0+cu130、vLLM 0.24.0、transformers 5.9.0、ray 2.55.1、datasets 5.0.0。这些记录是环境观察值，不是可移植的依赖锁。旧文件 `repro/archive-20260929/autodl-pip-freeze.txt` 来自 Conda base，不要据此安装。全新的 GPU 安装仍待核验。

## 4. 固定模型与数据集

```bash
export MODEL_PATH=/path/to/persistent-workspace/models/Qwen3.5-2B-15852e8
"$VERL_DIR/.venv/bin/python" scripts/download_model.py --output-dir "$MODEL_PATH"
# 对已有快照做离线完整性检查：
python3 scripts/download_model.py --verify-only --output-dir "$MODEL_PATH"
export DATA_DIR="$VERL_DIR/data/probegrpo-ragen-public-sokoban-v1"
"$VERL_DIR/.venv/bin/python" scripts/prepare_public_sokoban_data.py --output-dir "$DATA_DIR"
```

模型：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。下载脚本只使用该 revision，并检查记录中的 13 个文件哈希。权重文件共 4,548,221,488 字节。上游模型许可证和署名要求仍然适用。

数据集：`ZihanWang314/ragen-datasets`，revision `e2060cf7c51da891a68e0e7437309e5ec052e45b`。准备脚本会核对源文件 SHA-256、棋盘去重、排除训练/测试重叠，并计算精确最短解。默认训练集 512、测试集 128，最大 12 步。也可通过 `--source-dir /path/to/raw-ragen` 使用离线原始备份，目录内需有 `train.parquet` 和 `test.parquet`。原始数据的来源元信息和许可证仍适用；项目许可证不会重新授权第三方数据集。

在记录的训练包版本下，重建后的输出哈希为：

- train：`75ac2a084eba22d34d18ed214791e1775c71cb9cd739eca875d46402e224d736`
- test：`31d8b0b4ec5d8951af305289eb430686ed9e95a409fa0d43e0e6abd9d87b2728`

不同 parquet 库版本可能产生不同序列化字节；比较文件哈希时应使用记录的环境。不要静默替换数据集 revision。

## 5. GPU smoke，然后运行公开对比实验

历史运行使用一张 NVIDIA GeForce RTX 4090 48 GB。整理仓库时没有分配 GPU。本配方要求兼容的 48 GB GPU 和可用 CUDA 驱动；未证明其他 GPU/驱动兼容。

```bash
bash scripts/check_verl_stack.sh "$VERL_DIR"
OUTPUT_DIR="$PWD/outputs/grpo-gate-new" bash scripts/run_verl_grpo_smoke.sh "$VERL_DIR"
OUTPUT_DIR="$PWD/outputs/sokoban-gate-new" bash scripts/run_verl_sokoban_smoke.sh "$VERL_DIR"
# 检查 smoke 结果并确认磁盘空间后，再跑主实验。
SEED=17 RUN_TAG=preserved-v1 bash scripts/run_sokoban_ablation.sh "$VERL_DIR" main
# 将 SEED 依次改成 42 和 101，完成全部 12 次运行。
```

公开实验启动器使用 50 个 update、四种方法；probe 方法使用 B=2，并固定训练/测试输入。先阅读 `docs/SOKOBAN_AGENTLOOP.md` 和结果记录，了解不同 seed 的历史来源差异。当前源码是维护中的复现配方，不表示所有历史运行都使用同一个 commit。运行 manifest 会记录源码状态、框架/启动器哈希、相关依赖版本、checkpoint 频率和用户 Hydra 覆盖项；日志中含解析后的配置。

**Checkpoint 行为：**历史主实验设置 `trainer.save_freq=-1`，因此没有可下载的主结果最终权重。新启动器默认保存最终 step 的权重。这样不会改变 loss 或采样设置，但会增加磁盘用量。现存完整 FSDP smoke checkpoint 约 9.4 GB；若要连续保留 12 个相似大小的最终 checkpoint、运行环境、模型和日志，至少准备 150 GB 持久化存储，或每跑完一组就归档。原先的 100 GB 服务器无法容纳全部文件。设 `SAVE_FREQ=-1` 可以复刻历史的不保存 checkpoint 配置，但也会再次丢失最终权重。

归档中的早期 GSM8K step-6 checkpoint 不是最终 Sokoban 模型，也不是独立 LoRA adapter。若尝试恢复训练，模型、优化器、元信息和固定框架目录必须配套保留；干净环境恢复尚未核验。

## 已知缺项与后续核验

1. 在全新 Linux GPU 环境中安装、运行 smoke 并验证恢复训练。
2. 使用固定输入重新运行主协议并评估。
3. 主实验最终权重（历史从未保存，需要重训）。
4. 导出可移植的训练 adapter，并提供仅推理演示。

这些是明确缺口。CPU 检查成功、归档分数吻合，都不能代替上述验证。
