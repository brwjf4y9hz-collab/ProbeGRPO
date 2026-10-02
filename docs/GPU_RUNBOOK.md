# AutoDL GPU 运行手册

## 固定训练栈

ProbeGRPO 使用当前版 `verl` 训练，不安装 RAGEN 固定的训练栈。首次 GPU 里程碑采用：

- `verl` commit `cf14ded3a448107e70a206fd201817cc1cbae348`；
- 该 commit 的固定 `uv.lock`，使用 `vllm` 和 `fsdp` extras，再覆盖 NumPy 2.3.5；
- `Qwen/Qwen3.5-2B`；
- 至少 45,000 MiB 可用显存的一张 NVIDIA GPU；
- CUDA 驱动兼容 12.8 或更新版本；
- 至少 80 GiB 可用的工作区存储。

RAGEN 仅作为环境参考：标准 GRPO 栈通过后，再将其 Sokoban/WebShop 行为适配到当前 verl 的 `AgentLoop`。不要把 RAGEN 的 `vllm==0.8.2` 装进此运行环境。

## AutoDL 环境准备

选择一张 48 GB GPU，例如 RTX A6000、A40 或 L40S。项目仓库和运行环境都放在 AutoDL 持久数据盘目录下，不要放在容量较小的系统盘。以下命令中的路径需要改成实例上实际显示的路径。

```bash
cd /path/to/ProbeGRPO
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
bash scripts/check_verl_stack.sh /path/to/persistent-workspace/probegrpo-runtime/verl
```

Bootstrap 会固定依赖版本，并在 `runtime_manifest.txt` 记录 ProbeGRPO commit、verl commit、uv 版本和创建时间。若安装失败，保留完整命令输出；不要在原环境中单独升级 torch、Transformers、vLLM 或 verl。

锁文件固定 NumPy 2.4.6，与 Python 3.12 下的 mistral-common 1.11.3 不兼容。Bootstrap 会设置并记录 `numpy==2.3.5`，随后执行 `uv pip check`。再次运行 `uv sync --frozen` 会恢复不兼容版本，也可能移除 editable ProbeGRPO；若有意同步，必须重新覆盖 NumPy 版本并安装项目。上游锁文件本身不会被修改。

Bootstrap 只解析一次锁文件并创建 `verl/.venv`。后续命令会使用该环境的绝对 Python 路径（Ray worker 也一样），避免误用当前 Conda 环境导致训练栈悄悄改变。Git 只拉取固定 commit；uv/Hugging Face 缓存保存在运行环境的数据目录中，不占 AutoDL 的小系统盘。

若 Hugging Face 访问缓慢，可在运行脚本前于 shell 中显式设置可信镜像。不要把访问 token 写入仓库，或放进会被 shell history 保存的命令行参数。

## 五步标准 GRPO gate

首次付费 GPU 运行刻意不使用 Agent 环境，而是单独验证 Qwen3.5 模型加载、vLLM rollout、FSDP2 LoRA 更新、GRPO 分组（`K=4`）、checkpoint 及精确软件锁。

```bash
cd /path/to/ProbeGRPO
bash scripts/run_verl_grpo_smoke.sh \
  /path/to/persistent-workspace/probegrpo-runtime/verl
```

默认设置：

- 每次 update 取两个 GSM8K prompt，每个 prompt 生成四条 rollout；
- LoRA rank 32、alpha 64，自动发现 `all-linear` target；
- prompt 上限 256，response 上限 1024；
- actor PPO mini-batch 为 2（verl 乘以 `rollout.n=4` 后得到 8 条轨迹）；
- dataloader worker 数为 0；Ray 临时目录为短路径 `/tmp/pgr`；OMP 默认线程数为 1；
- optimizer 与参数 offload；
- 训练五步，在 step 5 保存 checkpoint；
- 仅控制台日志，不登录 W&B。

在相同输出目录中再次运行以验证恢复训练：

```bash
TOTAL_TRAINING_STEPS=6 bash scripts/run_verl_grpo_smoke.sh \
  /path/to/persistent-workspace/probegrpo-runtime/verl \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/absolute/path/to/ProbeGRPO/outputs/verl-grpo-smoke/checkpoints/global_step_5 \
  trainer.save_freq=1
```

第二条命令应从 step 5 恢复，并且只再训练一个 update。

## 验收清单

- `check_gpu_host.sh` 和 `check_verl_stack.sh` 无额外覆盖项即可通过。
- Qwen3.5 识别为模型类型 `qwen3_5`。
- 五个 update 完成，无 NaN、Inf、CUDA OOM、Ray worker 退出或 tokenizer 不匹配。
- 每个 prompt 恰好生成四个 rollout 样本。
- 存在 step-5 checkpoint，并能恢复到 step 6。
- 日志包含 reward、response 长度、actor loss、rollout 时间和 update 时间。
- 停止实例前，在 `experiments/environment/` 记录显存峰值和总墙钟时间。

只有通过此 gate 后，才继续接入并运行 Sokoban AgentLoop，再增加确定性重放和 probe hook。

操作员报告的证据在 `experiments/environment/2026-09-19-gpu-gate.md`。修正后的默认参数已通过单步信号 smoke；五步训练及恢复训练曾使用较早的 256 token / mini-batch 8 配置测试。下一步见 `docs/SOKOBAN_AGENTLOOP.md`。
