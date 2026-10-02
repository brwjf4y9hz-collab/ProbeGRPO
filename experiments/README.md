# 实验协议

## 实验方法

| ID | 方法 | Probe 预算 | 选点方式 |
|---|---|---:|---|
| `grpo` | 标准 GRPO | 0 | 无 |
| `random_b2` | 成本匹配的随机对照 | 2 | 分层随机 |
| `surprisal_b2` | 不确定性启发式 | 2 | chosen-token surprisal |
| `linear_ucb_b2` | ProbeGRPO | 2 | 学习每个 rollout token 的价值 |

使用种子 `17`、`42` 和 `101`。先在 seed `17` 上调参；其他两个 seed 开始前冻结所有选择。

四种配置都能在固定版本的当前 verl 栈上运行。正式实验使用校验和已核对的公开 RAGEN Sokoban 数据；训练/测试划分分别含 512/128 个去重棋盘，棋盘布局无交集，并用精确最短路径 oracle 验证 12-turn horizon。Tiny 手工关卡仅用于接线测试，不会进入结果表。

后端提供已采样动作 token 的 log probability，但不提供完整 next-token 分布。因此 `surprisal_b2` 明确命名和记录为 chosen-token surprisal，不将其误称为 entropy。LinearUCB 使用该可获取的不确定性统计量。当前 `B` 会在每个 prompt group 中先 probe 前 `B` 条 episode，再在每条 episode 中选一个 turn；整组 top-B 还未实现。

## 必须记录的元数据

每次运行需记录：

- ProbeGRPO 和 verl 的 Git revision；若适用，也记录 RAGEN 环境源码 revision；
- 模型 revision 与 tokenizer/chat-template 哈希；
- 完整解析后的 Hydra 配置；
- CUDA、驱动、torch、transformers、vLLM、verl 和 PEFT 版本；
- 随机种子和环境数据划分；
- 主 rollout/probe rollout token 数及 GPU 小时；
- 成功率、奖励、步数、非法动作、chosen-token surprisal、advantage 统计和重放不匹配率。

## 消融

主流程稳定后再执行：

- budget：`1`、`2`、`4`；
- lambda：`0.25`、`0.5`、`1.0`；
- token mask：完整 assistant turn 或解析出的 action span；
- scheduler：随机、chosen-token surprisal、LinearUCB。

不要把所有设置都乘以三个种子。先用一个 seed 选择配置，再用全部三个种子评估冻结后的优选项和 baseline。

一次只运行一个冻结 seed：

```bash
bash scripts/run_sokoban_ablation.sh /root/autodl-tmp/probegrpo-runtime/verl main
```

将 `SEED` 设为 17、42 或 101。运行器拒绝复用输出目录、记录代码/数据 revision，并在全部方法臂结束后生成 `summary.json` 和 `summary.md`。main-v1 已完成，精简记录在 [`results/public_sokoban_main_v1`](results/public_sokoban_main_v1)。

以下命令可重新生成汇总表和矢量图：

```bash
make results
```

按日期保存运行故障及修复方式。首次公开数据运行中的网络、CUDA 显存、同机初始化和合成 padding 事件见 [`environment/2026-09-21-public-ablation-incidents.md`](environment/2026-09-21-public-ablation-incidents.md)。

## 结果保存规范

机器可读原始记录放在 Git 忽略的 `outputs/*.jsonl` 中。Git 仅提交脱敏汇总、图、解析后的配置和含哈希的 manifest。负结果和崩溃运行也要留在表中。
