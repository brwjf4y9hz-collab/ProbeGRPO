# 200-step 对照的硬件配置与学校服务器入口

状态：代码与 CPU 回归测试已准备；**本轮未使用 GPU**。24GB 的配置是候选方案，
没有宣称能放下模型、速度更快或已完成分布式训练/恢复验证。

## 实验规格与执行配置分开

实验协议固定四组 × 三种子 × 200 updates，原 Qwen3.5-2B、LoRA rank32、512/128 split、
LR=3e-6、全局四个 prompts/update、K4/K6、B2、λ0/0.5 不变，response cap 1024 不变。
硬件 JSON 只能设置 GPU 分片、CPU offload、每次计算 token 分块、熵计算分块和推理并发等执行项；
代码对白名单外的项直接报错。减少并发/计算分块不是减少总 rollout，也不是截短序列。

| Profile | 占用卡数 | 执行方式 | 当前证据 |
|---|---:|---|---|
| `single_48gb` | 1 | 原单卡设置 | 旧 50-step 跑过同样的基本执行设置；新版仍需 smoke |
| `single_24gb` | 1 | 更小 entropy 分块、rollout 同时最多 2 条 | 仅候选，可能仍 OOM |
| `dual_24gb` | 2 | FSDP2 分片=2、rollout TP=2，并发最多 4 条 | 仅候选，需验证 Qwen3.5/LoRA/TP2/通信与显存 |

两张 24GB 不等价于显存可无条件共用的一张 48GB。也不保证双卡有两倍速度。
三者均保留 actor token budget=2048，可容纳 prompt1024 + response1024 的完整样本；
不能把该上限压到 1024 然后误称为无损分块。单卡低显存候选主要减少熵计算分块和推理并发，
未改变全局 batch 或序列长度。默认仍为 `single_48gb`，单卡配置不会被双卡配置替换。
梯度累积/并行顺序可能带来数值差异；每个正式矩阵使用同一执行 profile，不能把不同硬件的
收益当算法收益。GPU 型号/数量/显存、torch/CUDA 运行版本、profile 哈希随 manifest 记录。
CPU RAM、PCIe 拓扑、驱动、学校队列时间需在实际机器检查；旧日志中的主机总内存使用量不能当本作业的内存需求。

## 现在可做：纯计划，不需要 GPU

默认单卡入口保留，省略 `--hardware-profile` 也会选择 `single_48gb`：

```bash
python3 scripts/run_final_controls.py \
  --verl-dir /path/to/probegrpo-runtime/verl \
  --hardware-profile single_48gb
```

选择学校候选配置时，才把最后一个参数改成 `single_24gb` 或 `dual_24gb`。
默认只输出计划，不访问不存在的运行环境，不导入 torch，不创建训练输出。
`--preflight` 进一步核对已有 runtime、数据与模型缓存，但仍不加载模型、不占 GPU、不写 SwanLab。

## 以后有卡再做：只使用分配给自己的卡

优先使用学校作业调度器分配的 `CUDA_VISIBLE_DEVICES`。没有这一变量时，且已明确获得卡号 0 和 2
的使用权，才可在启动器加 `--gpu-devices 0,2`。已有可见卡范围时，启动器拒绝改成另一组卡。
没有自动抢占第三张卡、清理他人进程、自动降规格或跨配置续跑的行为。

下面命令仅说明未来流程，当前不执行。路径换成学校的实际路径；如果作业调度器已设置卡，
不要额外传 `--gpu-devices`。

```bash
VERL=/path/to/probegrpo-runtime/verl
PROTOCOL_FILE=experiments/protocols/public_sokoban_controls_v3_200.json

# 无卡/CPU 就可核对已有文件，不启动模型。
"$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" \
  --hardware-profile dual_24gb --preflight

# 获得两张独占卡、检查固定环境后，先做四组 smoke。
SWANLAB_ENABLED=1 SWANLAB_MODE=cloud \
"$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" \
  --hardware-profile dual_24gb --allow-experimental-profile \
  --phase smoke --execute

# 同一源码和硬件的四组 smoke 均通过，才放行正式 12 组。
SWANLAB_ENABLED=1 SWANLAB_MODE=cloud \
"$VERL/.venv/bin/python" scripts/run_final_controls.py \
  --verl-dir "$VERL" --protocol-file "$PROTOCOL_FILE" \
  --hardware-profile dual_24gb --allow-experimental-profile \
  --phase main --execute
```

单卡 24GB 只需选择 `single_24gb`，同样要求候选确认和 smoke；不能保证成功。
单卡 48GB 使用 `single_48gb`，不需要 `--allow-experimental-profile`。
可见卡数必须和 profile 精确一致，每卡显存达到要求、可用显存至少 90%；不满足就停止，
这不等于通过运行时 OOM/通信/恢复验证。90% 是保守启动门槛，不是资源权限隔离机制。
学校调度器/管理员决定资源授权，本工具不能代替学校的隔离或安全策略。

## 输出、验收与汇总

- 基础目录下追加 `single_48gb` / `single_24gb` / `dual_24gb`，再按 seed/method 保存，禁止混跑。
- GPU smoke 只说明四个一更新管线有完整结果；**不自动证明 200 步稳定或恢复成功**。
- `check_run_complete.py` 逐 update/eval 验证数量与有限值，双卡 checkpoint 两个 rank 都必须在；
  CSV 带验收哈希，失败类别进入 receipt，部分运行不会被自动覆盖。
- `summarize_final_controls.py <基础目录>/<profile>` 在 12 组验收完整时生成 JSON/Markdown 配对结果，
  拒绝混合执行 profile/不同 GPU 型号组合和验收后改动的指标。
- CPU单测使用结构fixture，不是低规格正式训练。学校SSH信息已由用户私下提供；登录授权/资源分配与部署另行核验，未声称已占卡或跑通。见[SCHOOL_GPU_RUNBOOK_ZH.md](SCHOOL_GPU_RUNBOOK_ZH.md)。

配置键来源：固定 verl commit 的
[rollout config](https://github.com/verl-project/verl/blob/cf14ded3a448107e70a206fd201817cc1cbae348/verl/trainer/config/rollout/rollout.yaml)
和已有可运行启动器；checkpoint rank 命名核对了固定版本的
[FSDP checkpoint manager](https://github.com/verl-project/verl/blob/cf14ded3a448107e70a206fd201817cc1cbae348/verl/utils/checkpoint/fsdp_checkpoint_manager.py)。
配置项存在不等于模型/硬件组合已验证。
