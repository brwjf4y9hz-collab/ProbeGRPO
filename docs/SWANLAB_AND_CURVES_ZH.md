# SwanLab 与训练曲线

ProbeGRPO 的 verl 训练入口支持可选 SwanLab 跟踪。默认仍只写 console 日志，不登录、不上传数据。开启后由 verl 原生 logger 按 step 记录训练指标；训练结束时还会从原始日志和 Sokoban episode sidecar 导出本地 CSV 与 SVG。

## 在 GPU 运行环境安装

在持久化的 verl 环境中安装 SwanLab：

```bash
uv pip install --python /path/to/probegrpo-runtime/verl/.venv/bin/python 'swanlab==0.10.2'
```

云端跟踪需要先在 SwanLab 创建 API Key，并通过环境变量提供；不要把 key 写进脚本、配置或提交记录：

```bash
export SWANLAB_API_KEY='你的 API Key'
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/path/to/persistent-workspace/swanlog
```

若先想本地检查数据、不上传云端，可用 `SWANLAB_MODE=local`。SwanLab 的模式还包括 `offline` 和 `disabled`；模式行为及登录方式以 [SwanLab verl 官方集成说明](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html) 为准。

## 启动

单次 GRPO smoke：

```bash
SWANLAB_ENABLED=1 bash scripts/run_verl_grpo_smoke.sh /path/to/probegrpo-runtime/verl \
  trainer.test_freq=1 \
  trainer.val_before_train=true
```

公开 Sokoban 对比：

```bash
SWANLAB_ENABLED=1 ROOT_OUTPUT=/path/to/new-output-root \
  bash scripts/run_sokoban_ablation.sh /path/to/probegrpo-runtime/verl main
```

每个正式 run 使用独立的 `trainer.experiment_name`，会出现在 SwanLab 的 ProbeGRPO 项目下。只有 `SWANLAB_ENABLED=1` 时才把 `swanlab` 加入 verl logger；启用但运行环境未安装包时，脚本会在训练前给出安装提示。未设置 API Key 时不要在无人值守任务中依赖交互式登录，可先用 local/offline 模式验证。

已保存的远端仓库也新增了兼容启动入口，不需要覆盖原有训练脚本：

```bash
cd /root/autodl-tmp/ProbeGRPO
SWANLAB_MODE=offline bash scripts/run_public_sokoban_swanlab.sh \
  /root/autodl-tmp/probegrpo-runtime/verl
```

登录后将 mode 改为 `cloud` 即可实时跟踪。这个入口会启动新模型训练；历史数据导入与同步不需要使用它。

## 本地曲线文件

训练脚本结束后，每个 run 输出目录下会有：

```text
curves/metrics_by_step.csv
curves/training_curves.svg
```

SVG 当前覆盖训练 reward、policy advantage、actor policy loss、gradient norm、训练/验证成功率、有效 probe 数和 probe token 成本。缺失的指标会明确显示 `not recorded`，不会伪造。CSV 保留日志中实际出现的全部逐步 scalar 指标，便于后续用 pandas、Excel 或绘图工具检查。

三种子主实验的原始日志已从远端恢复，并生成了可重新绘图的 compact CSV；详见 [历史曲线与证据](../experiments/results/public_sokoban_main_v1/training_history/README.md)。验证曲线来自原日志与验证 episode sidecar；验证频率由 `trainer.test_freq` 决定。需要更密的曲线时，应在新实验配置中调整验证频率，并记录额外生成成本。

单独重新导出（例如从 AutoDL 拷回旧 run 日志后）：

```bash
python3 scripts/plot_training_curves.py outputs/public-sokoban-grpo-seed17
```

## 历史实验已导入 SwanLab

全部 12 个历史实验现已同步到 [SwanLab ProbeGRPO 项目](https://swanlab.cn/@brwjf4y9hz-collab/probegrpo)。账号显示名为 ekko，实际用户名是 `brwjf4y9hz-collab`。这批记录已上传，日常查看直接打开链接即可。

2026-10-13 已在远端安装 SwanLab 0.10.2，导入全部 12 个原始 run；每个包含 51 个指标 step（含 step 0），并用独立的历史导入标签记录原始配置与日志哈希。导入目录：

```text
/root/autodl-tmp/ProbeGRPO/outputs/swanlab-history-20261013-v1
```

以后同步新的离线实验时，在远端终端运行（`NEW_RUN_DIRECTORY` 应替换为尚未上传的单个 `run-*` 目录）：

```bash
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/swanlab login
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/swanlab sync \
  --workspace brwjf4y9hz-collab --project probegrpo \
  /path/to/NEW_RUN_DIRECTORY
```

如果当前终端提示符仍是 Mac 本机的 `ekko@...`，可以直接执行下面这条命令完成远端登录：

```bash
ssh -t -p 57700 root@connect.westc.seetacloud.com \
  '/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/swanlab login'
```

`/root/autodl-tmp/...` 是服务器路径，不是 Mac 上的本地路径。API Key 在这条命令打开的终端提示中输入。

使用生成离线日志的相同 SDK 版本执行同步，参照 [官方离线同步说明](https://docs.swanlab.cn/guide_cloud/experiment_track/sync-logfile.html)。登录和同步不会重跑模型训练。

本次使用 SwanLab 0.10.2 的多目录、同进程同步时，第一组上传成功，第二组出现项目 READ 权限错误；新的进程能够正常读取项目并同步。最终改为每个目录独立启动一次 CLI 完成了剩余 11 组，同步后的云端项目共有 12 组。后续批量同步也应每组独立运行；避免对已上传的记录无条件重复同步，因为未指定云端 run ID 时 CLI 会创建新实验。

`scripts/audit_swanlab_history.py` 可以只读核对云端更新 step、actor policy loss、验证成功率、累计 rollout tokens 和逐步 probe tokens，与导出源 CSV 比较。核对结果保存在主实验 `training_history/swanlab_cloud_receipt.json`。

若需要 `swanlab watch` 本地看板，需额外安装 `swanlab[dashboard]`，并使用 local 模式生成记录；离线看板与可同步的 offline 记录是两个用途不同的模式，详见 [官方离线看板说明](https://docs.swanlab.cn/guide_cloud/offline-board.html)。
