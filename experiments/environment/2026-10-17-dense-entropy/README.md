# Dense entropy 修复验证（2026-10-17）

范围：修复固定verl dense分支未使用chunking的实现缺口；不是新算法或新GPU实验结果。

## 指纹

- verl commit：`cf14ded3a448107e70a206fd201817cc1cbae348`，不升级依赖。
- 原engine SHA256：`79d8779fb2a0b8a1ac7b3ec85fba2521ac8cae079925a497c4366fd77f8abfbe`。
- 修复后engine SHA256：`be9daa3ad741ad2539752344de01db3859bd580de6697b4719166a9c1b4dab97`。
- 新patch SHA256：`20925f05fed751502c9b1167dd0594823796da16ae271fc947568fcd475b0a73`。
- 历史trainer patch未修改；原始日志、旧checkpoint及历史结果未覆盖。

## 已执行

本地 `PYTHONPATH=src python3 -m unittest discover -s tests -q`：139项，132通过，7跳过。
`make check-scripts`、`git diff --check`、两项scripted CPU smoke通过。

在现有训练环境的**独立副本**先两次运行setup验证真实补丁应用与幂等，然后：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PROBEGRPO_TEST_VERL_DIR=/isolated/fixed-verl \
  PYTHONPATH=/isolated/source/src /existing/runtime/python \
  -m unittest discover -s /isolated/source/tests -v
```

上述是命令模板；真实隔离目录与工具输出见本任务执行记录，本地ignored outputs保留测试源码与验收摘要，
不在公开仓库发布学校主机信息。
该测试包仅含新增测试，真实结果为9项全部通过。实际patched engine的dense if分支经AST单独执行，
未初始化FSDP/Ray或加载模型；这不等同分布式forward端到端训练。

新旧熵函数使用相同dtype与公式，但只声明测试容差内数值/梯度一致，不保证CUDA逐位一致或轨迹相同。
forward-only单次熵调用上限为chunk_size token行，不含模型本身和完整logits常驻显存；GPU峰值待学校测量。

## 未执行与后续

学校source同步、GPU四组重新smoke和200步矩阵均待执行。旧seed17 GRPO完成回执保留为旧版本证据，
不能复制其receipt到新源码结果目录。新的源码指纹包括patch文件，新run manifest记录engine/patch hash。
保持四组×三种子×200、模型/数据/K/batch/LR/序列长度及三个硬件profile不变。
