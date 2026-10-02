# Sokoban AgentLoop 阶段

## 范围

`SokobanEnv` 是一个确定性的真实二维转移环境，包含三个手工制作的单箱关卡。这些关卡是**集成测试夹具**，不是 RAGEN benchmark、程序生成关卡或泛化实验。`TinySokobanEnv` 仍作为 probing core 的独立微型测试环境。

正式实验使用固定 revision `e2060cf7c51da891a68e0e7437309e5ec052e45b` 的公开 `ZihanWang314/ragen-datasets` Sokoban parquet 文件。数据准备脚本会核验两个源文件校验和，将发布的棋盘转换成可重放 task ID，保留排序后前 512 个唯一训练布局和 128 个唯一测试布局，并排除所有与训练集重复的测试棋盘。每个选中的单箱棋盘都有不超过 11 步的精确最短解，小于固定的 12-turn horizon。所有实验臂都使用稀疏的终局成功奖励。

Episode driver 不依赖训练框架。可选的 `SokobanAgentLoop` 继承固定版本 verl 的 `AgentLoopBase`，调用其 Continuous Token 方法，每个动作调用一次模型服务，并返回带环境 `reward_score` 的 `AgentLoopOutput`。不使用 LLM judge。

2026-09-19 已在 AutoDL 单步 GPU smoke 中验证真实模型生成和 actor 训练；CPU 测试使用脚本动作，不代表模型学习结果。

## 数据流

```text
task_id + env_seed -> reset -> 棋盘 observation
     -> Qwen tokens -> 解析单个动作 -> env.step -> 下一棋盘 -> ...
     -> Episode（tokens、masks、前后状态哈希、动作前缀、奖励）
     -> AgentLoopOutput -> verl TransferQueue -> 标准 GRPO 更新
```

Response 坐标同时包含 assistant token 和中间环境反馈，只有 assistant 位置的 mask 为 1。初始 prompt 坐标不属于 response。最终 response 以 assistant token 结束。若 context merge 改动了已记录的 assistant token，程序会直接失败，不会静默错位后续 probe credit。

格式错误的生成文本和被挡住的移动都会消耗一次动作尝试，但棋盘不变；由于 step count 会增加，完整状态哈希仍会变化。Episode 在成功、环境步数上限、token 预算耗尽或生成空文本时停止。动作 token 不会在截断后继续执行。

后端若提供，可读取所采样 token 的 logprob。但此适配器的后端契约不提供完整分布 entropy 或 top-two margin，因此可运行的不确定性 baseline 明确称为 chosen-token surprisal。LinearUCB 使用同一可用统计量并记录来源，不将 surprisal 冒称为完整 entropy。Random、Surprisal 和 LinearUCB 都可在每条被 probe 的 episode 内选择一个 turn。Trainer hook 已在真实单步 GPU gate 中验证，会在标准 GRPO advantage 计算后应用 credit。

## 按顺序运行

本机 CPU 检查（无需 torch 和下载）：

```bash
make check
make agent-smoke
```

将本地分支同步到 AutoDL，并在已有远端运行环境重新安装 editable ProbeGRPO 后，先执行实际 Qwen/verl tokenizer 检查。它只加载缓存 tokenizer，使用脚本动作，并测试真实的 `AgentLoopOutput.as_dict()` 转换。AutoDL 2 GB no-card 模式下，导入运行环境本身也可能耗尽内存；请使用内存充足的 CPU 或 GPU 实例。

```bash
export HF_HOME=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface
export OMP_NUM_THREADS=1
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/check_sokoban_tokenizer.py
```

该检查通过后，再执行一次真实模型/GRPO update：

```bash
bash scripts/run_verl_sokoban_smoke.sh /root/autodl-tmp/probegrpo-runtime/verl
```

默认输出为 `outputs/verl-sokoban-smoke/`。标准 verl JSONL 不含自定义轨迹字段，因此 adapter 还会在 `rollouts/episodes/step-N/` 下为每条 episode 单独写 sidecar。Sidecar 保留完整 response mask、动作前缀、observation 和完整状态哈希。文件使用哈希 ID 和排他创建模式；若复用 ID 再跑，必须换新的 OUTPUT_DIR。

```bash
PYTHONPATH=src python3 scripts/inspect_sokoban_episodes.py \
  outputs/verl-sokoban-smoke/rollouts/episodes/step-1
```

验收标准：2 个 prompt × 每个 4 条采样 episode，共 8 条轨迹；sidecar mask 精确覆盖 assistant token；所有动作前缀都能重放到相同状态；有限长度 episode 返回环境奖励。预训练模型不一定能解出 smoke 夹具，成功不是强制条件。即使 reward 相同，GRPO advantage 也可能为零。2026-09-19 远端 smoke 完成了一次真实 Qwen/verl GRPO update，写出 8 个 sidecar，并由 replay/mask 检查器全部通过；8 条夹具轨迹中有 1 条成功。这是接入检查，不是 held-out benchmark，也不是 ProbeGRPO 效果提升结果。

## Paired-suffix 调试 gate

先同步本地分支到 AutoDL。在加载模型权重前，打开 probe 开关运行脚本 tokenizer 检查。它使用脚本 `up` 动作测试相同的双分支编排，不是真实模型结果：

```bash
export HF_HOME=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
PROBEGRPO_DEBUG_PROBE=1 \
  /root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/check_sokoban_tokenizer.py
```

通过后，使用**新的输出目录**运行一个真实 update。`MODEL_PATH` 指向已下载快照；以下 revision 已在 2026-09-19 主机核验：

```bash
export MODEL_PATH=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
bash scripts/run_verl_sokoban_probe_smoke.sh /root/autodl-tmp/probegrpo-runtime/verl
python3 scripts/inspect_sokoban_probes.py \
  outputs/verl-sokoban-probe-smoke/rollouts/episodes/step-1
```

2026-09-19 的付费 GPU gate 生成 8 条 episode、2 个有效 probe，delta 分别为 1.0 和 0.0，额外 response token 共 304 个。该运行没有把 probe credit 应用到 actor update。Smoke 配置最多在每个 training prompt group 的 session 0 选一个 probe。非法动作、replay/context 不匹配或后缀失败都会产生零 credit 并记录 skip reason。正式 adapter 会统计成功和失败分支实际生成的模型 token。公开数据的同预算运行完成前，不应声称训练有收益。

## Trainer advantage gate

固定版本 verl v1 trainer 先计算标准 GRPO advantage，再将其作为嵌套 tensor 写回 TransferQueue。ProbeGRPO hook 在两步之间运行。它通过 trajectory ID 将 batch 行与哈希 episode sidecar 关联，检查完整 response mask，将有效 probe credit 打包为 `[N,A,T]` 和 `[N,A]` 张量，并只修改选中 assistant turn。sidecar 缺失或不匹配会停止 update，不会把 credit 错分配给其他 episode。Hook 支持每条 episode 一个 TransferQueue span，`budget=0..4`。

同步分支后，可在不加载 Qwen 权重的情况下，用已安装运行环境检查真实 episode：

```bash
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python \
  scripts/check_saved_probe_hook.py \
  outputs/verl-sokoban-probe-smoke-882ea33-run2/rollouts/episodes/step-1
```

脚本会核对 `budget=0` 和 `lambda=0` 时与 GRPO 完全一致，再输出 `lambda=0.5` 修改过的 token 索引。Probe delta 现在除以 batch 内 `max(1, 最大绝对有效 delta)`，不做均值中心化。对于 `[1.0, 0.0]`，归一化结果是 `[1.0, 0.0]`，所以零 delta anchor 不会得到 credit。2026-09-20 首次 GPU update 使用较早的中心化 z-score 并修改了四个 token；那只是接线 gate，不能与之后的实验结果比较。

若要跑一次训练 update，先在固定外部 checkout 中安装幂等 hook，再换一个新输出目录：

```bash
export OUTPUT_DIR=/root/autodl-tmp/ProbeGRPO/outputs/verl-sokoban-probe-train-smoke-1
bash scripts/run_verl_sokoban_probe_train_smoke.sh \
  /root/autodl-tmp/probegrpo-runtime/verl
```

安装器会核验准确 verl commit，并在旁边保存原 trainer 的 `.py.probegrpo.backup`。训练日志必须包含 `probe/valid`、`probe/changed_tokens` 和 `probe/credit_abs_sum`；仅仅完成 optimizer step 不能证明 credit 实际参与了更新。

## 公开数据同预算实验

AgentLoop 会在模型生成前读取 `probe.enabled`、`probe.budget` 和 `probe.scheduler`。`budget=0` 不运行 paired suffix，且精确保留 GRPO advantage。每个 prompt 采样四条 episode 时，`budget=B` 会 probe sessions `[0, B)`，再由配置的调度器在各个选中 episode 内选一个 turn。

设置 `lambda_coef=0` 会保留 GRPO advantage，但仍会运行并统计配置中的 probe，因此它不是零成本 baseline。旧的 `PROBEGRPO_DEBUG_PROBE=1` 路径在训练 probe 配置缺失时仍可用于仅 rollout 的诊断。目前不是在四条轨迹上执行全局 top-B，这项限制必须在报告里明确写出。

正式 runner 会下载并核验公开 release、写入运行 manifest、拒绝混用已有输出目录，并顺序运行四个主实验臂：

```bash
bash scripts/run_sokoban_ablation.sh \
  /root/autodl-tmp/probegrpo-runtime/verl main
```

首轮使用 seed 17，比较 GRPO、Random-B2、Surprisal-B2、LinearUCB-B2；模型、公开划分、主 rollout 数、horizon 和 probe budget 一致。脚本写入 `summary.json` 和 `summary.md`。主实验流程稳定后再运行 `lambda` 和 `budget` 阶段；固定最终配置后，再用 seed 42 和 101 重复。
