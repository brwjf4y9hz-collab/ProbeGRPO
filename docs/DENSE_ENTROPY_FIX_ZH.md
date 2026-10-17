# 2026-10-17：dense entropy 显存修复

## 故障与证据边界

学校用户日志确认：四组一步双卡 smoke 完成；GRPO 从 step1 恢复到 step2，两个 rank
加载 LoRA、optimizer、RNG、LR scheduler 后更新和保存。正式 v3 矩阵中 seed17 GRPO-K4
有完成回执；随后 GRPO-K6 在 `FSDPEngine.prepare_model_outputs` 的 dense entropy 分支
OOM：申请 5.77 GiB，仅剩 3.60 GiB。其余未完成，不是完整 12-run 结果。

配置已经启用 entropy chunking，但固定 verl 的 `use_remove_padding=False` 分支忽略该配置，
仍调用完整 logits 熵函数。上游 chunk 函数只接受二维输入，直接套在三维 `[batch, token, vocab]`
上并不正确。此次增加的 `patches/verl-dense-entropy.patch` 与历史 trainer patch 分开保存。

## 改动

- 启用分块时按 token 视图切块调用**原熵函数**；保留形状、dtype、device、token 顺序和梯度。
- 非连续输入不通过全量 reshape/contiguous 复制整个词表张量。空 token 输入保留形状。
- `entropy_checkpointing` 开关两条路径都使用同一选择后的函数；关闭分块时保持旧调用。
- 不切词表、不截断序列、不跳过熵、不改 loss，不改模型、数据、K、全局 batch、LR、步数或种子。
- 仅减少 forward-only 熵计算的临时张量峰值，不承诺消除所有 logits/训练激活的内存需求。
- 安装器严格校验固定 commit、原始/已修复源码 SHA256、补丁 SHA256；拒绝未知编辑，重复执行安全。
- `setup_verl.sh` 安装两个精确补丁；运行 manifest 记录 entropy patch/engine 指纹，源码指纹包括 patch 文件。

## 部署与验收

更新已批准的发布分支后，在原 runtime 执行 `bash scripts/setup_verl.sh /absolute/runtime/verl`，
无需重新 bootstrap、uv sync、下载模型或改变依赖。安装检查可单独运行：

```bash
python3 scripts/install_verl_dense_entropy.py /absolute/runtime/verl --check-only
```

保存旧完成回执、checkpoint、失败日志。新源码必须使用**新输出根目录**重新运行同规格四组
`--phase smoke`。通过后仍运行 v3 的四组×三种子×200；保留单卡与双卡三个 profile，不改协议。
不要复制旧回执来绕过源码指纹检查，也不要将旧源码的一条 baseline 自动拼进新源码矩阵。
旧 baseline 保留为旧执行版本证据；是否可比较需要显式兼容性审计，不能声称逐位等价。

数值回归检查 dense/chunk 输出、dtype、梯度、checkpoint 两种模式、非连续输入与每次最多
chunk_size 行。CPU 测试和补丁应用成功不等于学校 GPU/长轨迹 K6 通过；单步 smoke 也未覆盖
200-step 内所有批次，正式阶段仍需观察峰值与中止条件。SwanLab 按用户要求延期。

## 本轮已执行验证

- 本地完整单测139项：132通过、7项因缺torch跳过；shell语法与两项CPU scripted smoke通过。
- 已有训练环境的独立源码目录中，禁用GPU：新增9项测试全部通过，包括真实补丁引擎中
  dense 分支的AST执行（分块开/关×checkpoint开/关）、float64/float32/bfloat16输出和梯度、
  非连续输入共享storage、chunk行数上限、空输入，以及安装器拒绝未知编辑/错误版本/补丁。
- 独立固定verl副本上两个补丁实际应用成功，再次setup识别already installed；原始环境未改动。
- 尚未部署学校checkout，未重跑学校GPU，未测到新版本显存峰值；不得把以上CPU验证写成K6长跑修复成功。
