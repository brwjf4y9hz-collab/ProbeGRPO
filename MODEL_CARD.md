# ProbeGRPO Adapter 模型卡

此文件用于记录由 ProbeGRPO 训练出的 LoRA adapter。发布 adapter 前请补全准确资料。

## 基础模型

- 默认模型：`Qwen/Qwen3.5-2B`
- 扩展模型：`Qwen/Qwen3.5-4B`

## 训练信息

填写基础模型准确 revision、数据/环境划分、随机种子、update 数、rollout token 总量、硬件、解析后的完整配置和 ProbeGRPO commit。

## 预期用途

用于隔离运行的 Sokoban 和 WebShop 环境中的研究及作品集演示。

## 局限

此模型不适用于真实购物或未隔离的网页操作。环境奖励可能不完整，并可能诱发取巧行为。反事实估计是局部且有噪声的，并且仅适用于可枚举的合法动作。

## 评估要求

所有 baseline 应使用相同主 rollout 和 probe 预算。报告所有 seed 与失败运行；不要使用测试集挑选最佳 checkpoint。
