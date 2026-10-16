# 无卡求职交付准备验收（2026-10-16）

本轮未开卡、未加载模型、未创建训练或SwanLab实验；原v3仍四组×三种子×200steps。
未修改训练算法/协议/hardware profiles，默认48GB单卡与两个24GB候选保留。

## 交付内容

- `docs/PROJECT_REVIEW_ZH.md`：问题、真实案例、三次故障、反直觉结果和证据边界的连贯复盘。
- `docs/ARCHITECTURE.md`：真实GPU调用链、唯一advantage接入点与自研/上游边界。
- `docs/RELATED_WORK_ZH.md`：GiGPO差异、分组定义和未来适配条件；没有声称已实现或优于GiGPO。
- `docs/INTERVIEW_GUIDE_ZH.md`：讲解提纲、追问与简历表述边界。
- `scripts/audit_state_groups.py`：完整轨迹状态分组审计，不改训练信号。
- `scripts/audit_engineering_logs.py`：从12组旧log提取内存与step时间，非优化前后对照。
- `scripts/check_clean_cpu.py`、`docs/RELEASE_CHECK_ZH.md`：隔离本地clone检查与公开版本门槛。
- CI新增完整CPU检查和committed clone检查；尚未推送，因此没有声称GitHub Actions已通过。

## 实测与证据

1. 本地当前源码隔离快照，clone前后Git status干净；两个CPU smoke、完整`make check`、三种硬件纯计划通过。
   共115项单测，113通过、2因未装PyTorch跳过。详细日志/receipt来自
   `outputs/clean-cpu-draft-20261016-v4/`。临时快照commit不是公开正式提交。
2. `ai-research-reproduction`在该隔离clone执行README的`bash scripts/smoke_test.sh`，
   标准证据保存在根`repro_outputs/`（ignored）；`--verify-output`返回`evidence_valid=true`，
   README round-trip和源码前后指纹均通过。未提供数值目标，因此`result_match=not_evaluated`，不称论文复现。
3. 技能最初将Quick-start段落的CPU命令排除为setup，首选GPU检查；没有执行该选择。
   README将现有CPU脚本放入明确的correctness章节后，重新规划、审阅并执行唯一CPU目标。
   复现目标clone内没有修补源码；原仓库的工程/文档修改属于用户授权的交付工作。
4. 原SSH恢复。独立目录
   `/root/autodl-tmp/ProbeGRPO/outputs/portfolio-prep-20261016-v1/source`
   用固定verl Python执行115项测试，全通过（含两个PyTorch张量测试）；显式隐藏GPU。
   两个CPU smoke与默认单卡200-step纯计划通过。执行文件哈希与本地一致：
   `98a0d4e8af23fcba90f232173179478537b4a73ded923fd3b76d095c53274cf8`。
   原服务器工作区未覆盖，原editable安装和trainer patch未改；这不是已切换生产运行目录或GPU验收。
   远端README是同轮较早的排版版，代码/配置哈希一致；本地最终README另经隔离clone验证。
5. 同一服务器CPU审计GRPO/UCB×三个种子的4800条完整训练episode、45169个动作。
   结果及来源哈希见 `experiments/results/public_sokoban_main_v1/state_group_audit/`。
   它支持“存在自然分组机会”，不支持GiGPO收益、等成本胜出或稳定性机制结论。
6. 历史12组运行内存和时延摘要见同级结果目录`engineering_audit/`。失败的11.97GiB申请不是峰值；
   CPU主机总使用量不是本作业RAM需求；没有生成未知吞吐提升数字。

## 保留的未完成项

- 原仓库有多批未提交改动；尚未提交、推送或在公开GitHub commit上clone验证。
- 原远端工作区仍是旧版本加既有未提交修改；本轮新增代码仅隔离存放和CPU验收，未覆盖中文改动。
- 无完整200-step新结果、24GB显存实测或新版checkpoint恢复结果；GiGPO正式适配与训练未启动。
- 下一次GPU之前核对并备份实际工作区，确认部署版本及runtime指向，不能直接假设隔离测试等于部署完成。

先前失败的本地隔离验收v1日志保留：Git的macOS临时目录警告在stderr，被误当作dirty status；
工具已分离stdout/stderr，后续完整验收通过，没有忽略真实Git状态变化。
