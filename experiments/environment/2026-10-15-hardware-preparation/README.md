# 单卡保留与候选硬件入口：本地准备记录

日期：2026-10-15。只完成代码、配置和 CPU 检查，没有启动 GPU 训练、同步远端或创建 SwanLab 实验。

## 保持的实验规格

协议仍为 `public_sokoban_controls_v3_200`：四组 × 三种子 × 200 updates，共 12 runs、2400 updates。
模型、数据、学习率、全局 batch、序列长度、K4/K6、probe B2、λ 和 warm-up 不因硬件而缩小。

- `single_48gb`：默认且保留的单卡执行配置，沿用历史基本执行设置；新版仍需四组 GPU smoke。
- `single_24gb`：额外保留的单卡候选，仅减少 entropy 计算分块和 rollout 并发，可能仍 OOM。
- `dual_24gb`：双卡分片/TP 候选，不替换单卡入口；不保证模型兼容、显存可用或两倍速度。

三者输出按 profile 隔离。启动器不覆盖调度器分配的可见卡范围，不自动取用其他卡；
正式训练要求同源码、同硬件组合的四组 smoke 通过。验收与汇总拒绝混合硬件、缺失 rank checkpoint、
验收后被修改的指标，以及未覆盖 0–200 全历史的正式结果。

## 本地证据

运行 `make check`：共 104 项测试，102 项通过，2 项因本地没有 PyTorch 跳过；
CPU data-flow smoke、AgentLoop smoke、shell 语法检查通过。`git diff --check` 通过。
原始日志见同目录 `local-check.log`。跳过的张量测试需要以后在固定 verl runtime 重验。

三个 profile 的纯计划均在不存在的 runtime 路径下成功输出，各 12 runs、2400 updates，
说明计划模式不依赖 GPU 或实际运行环境。计划 JSON 随本记录保存。
测试中的 GPU inventory、checkpoint 文件和训练历史是明确的 CPU 结构 fixture，不是真实训练产物。

本记录不替代 GPU smoke、完整 200-step 矩阵、模型保存/恢复验证或学校机器上的实测显存和时延。
本轮改动尚未提交、推送或部署；旧 50-step 结果与早期环境验收记录保留，不作为新版运行证据。

使用说明见 `docs/HARDWARE_PROFILES_ZH.md`；简历六条标准的证据与缺口见
`docs/PORTFOLIO_ACCEPTANCE_ZH.md`。
