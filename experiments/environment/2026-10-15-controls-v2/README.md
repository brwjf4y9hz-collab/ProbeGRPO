# v2 无卡准备验收：2026-10-15

本记录是代码/配置/历史证据验收，**不是新 GPU 训练或新性能结果**。

## 固定规格与授权边界

用户要求补对照、修复并完善项目；明确要求需要 GPU 就说明，不因无卡缩减规格。
保持 Qwen3.5-2B、原 immutable 模型/verl/数据版本、512 train / 128 test、50 updates、
seed17/42/101。准备 GRPO-K4、GRPO-K6、UCB-B2 λ=0、修复 UCB-B2 λ=0.5 共 12 runs。
一更新 smoke 是独立检查，不能替代正式矩阵。150-step 四组 seed17 为可选后续扩展，不自动启动。

## 实际完成

- 本地 `make check`：90 tests，因本地无 torch 跳过 2 个张量测试，其余通过；CPU smoke、
  AgentLoop smoke 和全部 shell 语法检查通过。
- 远端固定 `.venv/bin/python`：**90 tests 全部通过、无跳过**。原始输出见 `tests.log`。
- 真实固定 verl 的 Hydra composition 接受 `K=6`、mini-batch 4、最终 checkpoint、
  `save_lora_only=true`。见 `hydra.log`；这不是模型保存/恢复的实际执行。
- 无卡 preflight 核对：固定 verl commit、512/128 派生文件校验和、数据 horizon/revision、
  本地 cached 模型 config/分片存在、editable 包指向当前远端仓库。见 `preflight.txt`。
- 远端最终执行 `torch.cuda.is_available()` 返回 False；未创建 `outputs/final-controls-v2`。
  无模型训练、未上传新正式 SwanLab run。
- v1 原始轨迹审计：612 warm-up anchors，horizon buckets 610/2/0；445 次同时有三桶候选。
  三种子首次真实 UCB 排名为 step28。结果和来源哈希已保存到 v1 `anchor_audit/`。
- 从 v1 seed17 按确定性规则提取正/负/零三个案例，检查 replay state/observation、
  factual 单步转移和 assistant token mask。见 v1 `trajectory_demo/`。未重生成两条模型 suffix。

## 部署与可恢复性

远端仓库：`/root/autodl-tmp/ProbeGRPO`，部署前 HEAD `d390b87`。
保留已有中文注释和 `setup_verl.sh` 精确补丁校验；没有用本地较旧 HEAD 重置远端仓库。
被修改的原有核心文件、测试和启动器已备份到
`outputs/controls-v2-before-20261015.tar.gz`。没有删除旧结果或改写 v1 数字。

最终远端可执行输入组合 SHA-256：
`cf709c3416901d045ff3a95019d66a427aad59985ec18d15e3fc1c114c608ec1`。
指纹覆盖 src/scripts/configs/协议及 pyproject，不包括输出和文档；源代码有未提交修改时，
新 manifest 仍会记录真实输入，而不只写 Git HEAD。此次没有创建 Git 提交或推送。

## 仍待有卡验收

需要原规格的一张 48GB GPU：检查 K6 显存与 padding、真实 LoRA-only 保存/恢复、
SwanLab 实时记录和四组一更新 smoke。随后执行正式 12 runs，保持原规格。
约 19.74 GiB 剩余磁盘，使用原生 LoRA-only checkpoint 加 optimizer/extra；不足时停止并
保留成果，不能靠少跑种子/步数或删除旧实验规避。实际新 checkpoint 尺寸仍需 GPU gate 核对。

K6 不预称等算力，运行后按实际生成 token 和时间报告。完整结果汇总拒绝混用 source snapshot、
缺失实验或未验收运行；不因测试通过就宣称效果提高或因果关系成立。
