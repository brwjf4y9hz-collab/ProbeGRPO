# 求职交付：隔离复现与公开发布检查

当前任务不启 GPU，也不为了通过检查修改正式实验协议。

## 1. 验证当前未提交源码，不污染原仓库

```bash
python3 scripts/check_clean_cpu.py --mode working-tree \
  --output-dir outputs/clean-cpu-draft-new
```

工具从源码白名单生成临时本地 Git snapshot，再独立 clone；不复制 outputs、模型、.env、
私钥或原仓库 Git 配置，不修改原 index/分支。子进程不继承 API Key、PYTHONPATH 或 GPU 可见性。
运行两个 CPU smoke、`make check` 和三种硬件 profile 的纯计划；前后 Git status 必须干净。
receipt 保存逐文件哈希、命令、日志和快照 commit。测试中的 fixture 不是真训练。
白名单过滤不是完整秘密扫描，不替代人工检查源码/日志中的凭据。

这个结果只能叫“当前源码隔离验证”，不能叫“公开仓库已可复现”。临时 snapshot commit 不属于原仓库历史，
不推送它，不将它写成正式实验的源码 revision。

## 2. 提交前审查，不能直接 git add 全部

先审查 `git status --short` 与 diff，确认用户已有改动的归属和提交边界。排除 `.DS_Store`、
AppleDouble `._*`、本地 runtime、原始日志、token、checkpoint、API Key 和个人路径。
实验 CSV/审计只提交去敏后的必要摘要与来源哈希，完整原始 trace 按已有策略留在 ignored outputs。

本轮代码与文档跨越既有多批修改，应按“实验正确性/协议、跟踪与证据、求职文档”分组审查，
不能把全部 dirty files 不加区分地作为一次本轮提交，也不重写既有历史。

## 3. 提交后验证真实 commit

```bash
python3 scripts/check_clean_cpu.py --mode committed \
  --output-dir outputs/clean-cpu-committed-new
```

默认 `committed` 只 clone 当前 HEAD，不包含未提交文件。旧 HEAD 缺新入口时应失败，不应复制文件进去伪造通过。
需要 Git、make、Python；完整本地验收使用 Python3.12，CI 另检查3.9/3.12。
它不联网安装 torch，不验证 GPU 的 Hydra compose、模型加载或 LoRA 恢复。

## 4. 推送后才算公开版本验收

推送需用户明确授权并确认分支/目标；从 GitHub 全新 clone 目标 commit 后，运行 README CPU 命令和
上述 committed 验证，记录远端 URL/commit、Python、命令、跳过测试和日志哈希。
不能用本地 clone 或“推送命令返回成功”替代公开 clone 验收。CI 有无通过也分别记录。

## 5. GPU 门槛留到开卡之后

原 v3 四组 × 三种子 × 200 steps、单卡默认和候选 profile 均不变。先固定 runtime 检查、
四组 smoke、真实 checkpoint 保存/恢复，再正式矩阵；新结果不与旧50步合并。
无卡阶段只做数据/源码/缓存检查，不能把 CPU 通过写成24GB显存适配成功。
