# ProbeGRPO

**多轮 Agent 的预算约束反事实探测与局部信用分配。**

ProbeGRPO 在 GRPO 训练链路中重放选中的决策状态，对比原动作与替代动作的后续奖励，将局部奖励差用于对应 assistant turn 的策略更新。项目包含确定性环境回放、探测调度、token mask、verl 接入，以及公开 Sokoban 的多种子训练与成本分析。

| 阅读目的 | 入口 |
|---|---|
| 快速了解问题与方法 | [项目导览](docs/PROJECT_WALKTHROUGH.md) |
| 查看结果图与逐种子数据 | [结果分析](docs/results/README.md) |
| 查看回放、选点与训练接入 | [系统架构](docs/ARCHITECTURE.md) |
| 运行演示或准备 GPU 实验 | [快速开始](#快速开始) · [完整复现说明](REPRODUCE.md) |
| 阅读新增复盘与面试材料 | [工程复盘](docs/PROJECT_REVIEW_ZH.md) · [面试提纲](docs/INTERVIEW_GUIDE_ZH.md) |
| 理解近邻方法与实验缺口 | [GiGPO 对比](docs/RELATED_WORK_ZH.md) · [求职验收](docs/PORTFOLIO_ACCEPTANCE_ZH.md) |

## 当前准备：修复后的 200-step 对照（尚未训练）

历史 v1 的50-step结果保留。新协议为 GRPO-K4、GRPO-K6、probe-only λ=0 与完整 ProbeGRPO
四组 × 三种子 × 200 updates；主终点固定step200。K6是更多主rollout对照，不自动等算力。
详见[对照协议](docs/FINAL_CONTROLS_ZH.md)。默认只输出计划，不启动模型：

```bash
python3 scripts/run_final_controls.py --verl-dir /path/to/verl \
  --protocol-file experiments/protocols/public_sokoban_controls_v3_200.json
```

保留48GB单卡默认；24GB单卡/双卡是待GPU验证的独立候选，不缩减模型、数据、batch或训练步数。
见[硬件入口](docs/HARDWARE_PROFILES_ZH.md)。

## 实验结果

模型为 `Qwen/Qwen3.5-2B`；512 个训练关卡、128 个共同留出关卡。四种方法各运行三个种子（17、42、101）、每次 50 次训练更新。下表为跨种子均值 ± 样本标准差；百分点简称 pp。

| 方法 | 留出成功率 | 相对 GRPO 的配对差值 | 额外 rollout token |
|---|---:|---:|---:|
| GRPO | 24.0% ± 2.0 pp | — | 0.0% |
| Random-B2 | 29.2% ± 4.6 pp | +5.2 ± 6.5 pp | 51.2% |
| Surprisal-B2 | 31.2% ± 3.4 pp | +7.3 ± 4.3 pp | 57.3% |
| LinearUCB-B2 | 31.0% ± 1.2 pp | +7.0 ± 0.8 pp | 51.6% |

![ProbeGRPO 成功率、种子波动与探测成本](docs/results/figures/01_probegrpo.png)

三次历史运行中，LinearUCB-B2 均高于对应 GRPO；Surprisal 的成功率均值更高，Random 的每千额外 token 高影响 anchor 诊断更高。结果尚未证明某个调度器全面最优。

**结果范围：**seed17 GRPO 存在历史 padding 配置差异；比较没有匹配总 token 预算。额外 token 比例以各方法自身的主 rollout token 为分母。误差线是样本标准差，不是置信区间；12 次运行评估的是同一批 128 个关卡。主实验最终权重未保存，轨迹与评估记录已有归档。

[完整分析](docs/results/README.md) · [实验协议](experiments/results/public_sokoban_main_v1/README.md) · [归档核验](repro/validation/README.md)

原12组训练已恢复600次更新和72次验证的[逐步曲线与来源](experiments/results/public_sokoban_main_v1/training_history/README.md)，
支持 `make curves` 离线重画；依赖为可选 `plots` extra，不影响CPU核心。
[SwanLab](docs/SWANLAB_AND_CURVES_ZH.md) 为显式开启，不默认登录/上传。
新增[正负零真实轨迹案例](experiments/results/public_sokoban_main_v1/trajectory_demo/README.md)、
[warm-up审计](experiments/results/public_sokoban_main_v1/anchor_audit/README.md)、
[六组状态重用审计](experiments/results/public_sokoban_main_v1/state_group_audit/README.md)和
[历史内存/时延摘要](experiments/results/public_sokoban_main_v1/engineering_audit/summary.md)。
这些是旧记录的审计，不是新训练、GiGPO对照或credit稳定性机制证明。

## 方法与训练链路

```mermaid
flowchart TD
    A[多轮 rollout 与轨迹记录] --> B[预算约束 anchor 选点]
    B --> C[重放共同前缀并核验状态]
    C --> D[事实与反事实后续采样]
    D --> E[计算局部奖励差]
    E --> F[对齐 assistant turn token mask]
    F --> G[修正 GRPO advantage]
    G --> H[verl actor 更新与留出评估]
```

- **回放与探测：**`ReplayableEnv`、状态哈希和成对后续采样；回放不一致时跳过探测。
- **调度：**Random、chosen-token Surprisal、LinearUCB；记录探测信号与额外开销。
- **训练接入：**修正所选 assistant turn 的 advantage，零差值保持零，零预算退化为普通 GRPO。
- **环境与执行：**公开 RAGEN 数据适配、Sokoban AgentLoop、LoRA／FSDP2／vLLM 链路。

当前 B=2 实现先选择两条 episode，再分别选择 turn；尚未实现组内所有候选的全局 top-B。局部奖励差也不等于完整因果贡献。

## 快速开始

CPU 演示需要 Python 3.9+，核心包依赖标准库，无需 GPU、模型下载或 API Key。在仓库根目录执行：

```bash
bash scripts/run_cpu_smoke.sh
```

脚本验证核心探测与 advantage 数据流，以及微型 Agent 环境交互。核心 smoke 的成功标志为 `ProbeGRPO CPU data-flow smoke test passed`；这些演示不是正式训练成绩。

需要安装包时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make check
```

GPU 训练使用独立 NVIDIA Linux 环境。按[完整复现说明](REPRODUCE.md)、[GPU 手册](docs/GPU_RUNBOOK.md)及 [AgentLoop 手册](docs/SOKOBAN_AGENTLOOP.md)准备固定框架、数据和模型；CPU 演示不会触发这些步骤。

## CPU 正确性与发布验收

不安装训练依赖，也可以运行单测与CPU数据流检查：

```bash
bash scripts/smoke_test.sh
```

提交后可用独立本地clone验证全部CPU检查和三种硬件的纯计划：

```bash
python3 scripts/check_clean_cpu.py --mode committed --output-dir outputs/clean-cpu-new
```

`--mode working-tree` 只验证未提交源码快照，不能冒充公开GitHub版本；见[发布检查](docs/RELEASE_CHECK_ZH.md)。
缺少torch时张量测试明确跳过；CPU通过不证明GPU、24GB配置或checkpoint恢复成功。

历史问题ZIP保持不变，用`python3 scripts/verify_incident_archive.py`核验其32份文件和8项问题记录。
原`docs/incidents/verify_bundle.py`针对旧快照的live文件哈希；新源码演进后出现差异不等于ZIP损坏，不能重写旧manifest伪造一致。
学校新机器的登录、只读资源检查和GPU启动门槛见[学校部署手册](docs/SCHOOL_GPU_RUNBOOK_ZH.md)。

## 代码与证据

| 模块 | 位置 |
|---|---|
| 类型、回放与状态校验 | [types.py](src/probegrpo/types.py) · [replay.py](src/probegrpo/replay.py) |
| 探测与选点 | [probing.py](src/probegrpo/probing.py) · [episode_probing.py](src/probegrpo/episode_probing.py) · [schedulers.py](src/probegrpo/schedulers.py) |
| 局部 credit 与批次对齐 | [advantage.py](src/probegrpo/advantage.py) · [verl sidecar](src/probegrpo/integration/verl_v1_sidecar.py) |
| 多轮环境接入 | [Sokoban AgentLoop](src/probegrpo/integration/sokoban_agent_loop.py) |
| 正式实验与数据 | [公开 Sokoban 结果](experiments/results/public_sokoban_main_v1/) |
| 故障、约束与负结果 | [问题复盘](docs/incidents/README.md) |
| 固定版本与外部归档 | [保存清单](repro/ARTIFACTS.md) · [仓库盘点](REPO_AUDIT.md) |

项目实现与记录用于研究和复现，不声称反事实信用分配本身是原创方法。相关运行与评测项目为 [SWE-HarnessLab](https://github.com/brwjf4y9hz-collab/SWE-HarnessLab)；两个项目各自报告任务与成绩。
