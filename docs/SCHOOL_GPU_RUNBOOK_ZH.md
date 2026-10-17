# 学校服务器：先部署、验收，再开始200步训练

这是新机器的部署流程，不是已运行记录。所有真实主机地址、账号、资源授权和输出路径记录在本地
ignored outputs，不把学校内网信息或凭据写进公开配置。

## 1. 登录与资源授权先完成

在本机终端登录实际账号；包含`@`的用户名使用`ssh -l 'your-user@realm' YOUR_HOST`。
密码/API Key只在终端交互，不发到聊天或脚本。如果登录提示home不存在，应请管理员修复对应home
目录及权限，再用`ssh-copy-id`配置密钥；不要更改sshd配置、放宽目录权限或要求管理员公开密码。

确认分配给任务的具体GPU卡号、时长、CPU/RAM和磁盘配额。`nvidia-smi`空闲不等于允许占用；
只使用分配的卡，遵守学校调度器的`CUDA_VISIBLE_DEVICES`。不升级系统/驱动、不重启、不清理他人进程。

## 2. 只读检查，不加载模型

在已分配、可写的工作目录中clone明确的发布分支/commit，而不是依赖main尚未合并的内容。
仓库到位后可执行：

```bash
bash scripts/check_school_host.sh /absolute/assigned/workspace
bash scripts/run_cpu_smoke.sh
make check
```

保存GPU型号/显存/驱动、内存、剩余磁盘、PCIe拓扑和工具版本；尚未配置密钥不等于机器不可用。
`nvidia-smi`顶部的CUDA Version表示驱动支持的CUDA版本上限，不是已安装的torch/vLLM CUDA runtime。
不能看见CUDA13就自行改固定依赖版本。CPU总使用量也不是本作业的内存需求。

## 3. 保持原科学规格，按固定环境准备

先读`REPRODUCE.md`、`repro/ARTIFACTS.md`和`docs/GPU_RUNBOOK.md`。在自己的工作区准备固定verl commit、
精确patch、独立Python环境、固定模型revision与512/128数据，不使用系统Python安装训练依赖。
有合规访问权限时，可从现有归档校验恢复模型/数据/环境；只读校验哈希，不把API Key和整个用户配置复制过来。
环境不兼容应单独处理并记录，不用缩序列、量化模型、减数据或种子来制造“跑通”。

依赖/模型下载与安装不自动等同GPU训练。实际准备前检查磁盘与网络，并将UV/cache目录放在分配的持久盘，
避免缺失home或小系统盘导致半途失败。

### 代码与大文件分开

个人代码目录只保存源码、配置和轻量记录。模型权重、Python训练环境、下载缓存、轨迹、
Ray临时文件及checkpoint放在另行确认有权限、有配额的大文件目录，不能默认写到仓库或home。
`df`显示文件系统余量，不等于个人配额；还需核对学校配额/保留策略。路径未知时先只读检查
挂载和个人配额，无法确定则请管理员确认，不先下载，也不挪用其他人的目录。

部署时使用真实批准路径设置`HF_HOME`、`UV_CACHE_DIR`、`PIP_CACHE_DIR`、`TMPDIR`和
`RAY_TMPDIR`；独立verl及其`.venv`也放在大文件目录。实际启动器是否继承每个目录设置须验收，
不能只设置变量便声称全部输出已迁移。每次运行显式传`--output-root /approved/large-storage/controls-v3-200`，
它同时迁移启动器的运行输出与checkpoint根目录，仍会追加硬件profile子目录。
不得仅依赖`.gitignore`：它防止提交，不限制磁盘占用。未知空间时不自动删checkpoint、日志或缓存，
也不以减少训练步数/种子/对照组替代存储准备。

### ensurepip recovery：不改系统包，继续失败的安装

Ubuntu系统Python若缺`ensurepip`，首次bootstrap可能已准备好固定verl源码和补丁，但尚未安装uv及训练依赖。
这不是CUDA错误，也不能因为源码校验通过便记录为环境安装成功。不必使用sudo安装系统包；可以安装
[固定版本的官方独立uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)，
并用[unmanaged模式](https://docs.astral.sh/uv/reference/installer/#unmanaged-installations)避免修改shell配置。

先更新到包含此修复的发布分支。下面的`RUNTIME`必须替换为已批准、确认为本项目失败安装的绝对路径：

```bash
(
set -e
RUNTIME=/approved/large-storage/probegrpo-runtime
test -d "$RUNTIME/verl/.git"
curl -fL --retry 3 https://astral.sh/uv/0.12.15/install.sh \
  -o "$RUNTIME/uv-install-0.12.15.sh"
UV_UNMANAGED_INSTALL="$RUNTIME/bootstrap/bin" sh "$RUNTIME/uv-install-0.12.15.sh"
"$RUNTIME/bootstrap/bin/uv" --version
bash scripts/bootstrap_verl.sh "$RUNTIME" --resume
)
```

`--resume`仅继续**未完成的环境安装**，不是训练checkpoint恢复：保留现有文件，重新核验固定verl与精确补丁，
再按同一锁文件安装。它拒绝缺失/无框架目录、根符号链接、已有完成manifest或错误本地uv版本；默认仍拒绝覆盖。
缓存、managed Python、可执行文件链接及pip缓存默认位于runtime，不需要改`HOME`。
版本由bootstrap检查：`uv 0.12.15 (x86_64-unknown-linux-gnu)`等括号内平台/构建信息允许存在，
但不接受`0.12.150`、预发布后缀或其他uv版本；不要再对整行做相等比较。
不要删除失败目录或反复运行旧版脚本；若再次失败，保留输出定位，不擅自升级训练依赖。
本地shell流程测试用模拟工具，不证明学校Linux实际安装完成；以真实`Runtime prepared`及后续依赖检查为准。

## 4. 选择执行profile，不改实验矩阵

- `single_48gb`始终保留为默认；不能在24GB卡上误用它然后降低显存检查门槛。
- `single_24gb`是单卡低显存候选，可能仍OOM。
- `dual_24gb`是两卡FSDP/TP候选；只有明确获配两卡后才能尝试，不代表两卡显存无条件合并。
- 四卡机器不代表会用四卡；当前没有经过验证的四卡profile，不自动新增或抢占全部可见卡。

先运行不占卡的计划与文件preflight：

```bash
python3 scripts/run_final_controls.py --verl-dir /absolute/runtime/verl \
  --protocol-file experiments/protocols/public_sokoban_controls_v3_200.json \
  --hardware-profile dual_24gb
```

真实`--preflight`需要固定runtime、缓存和正确的ProbeGRPO源码指向，路径和CPU测试先核对。
示例profile不代表已获两卡授权；获得分配后按`HARDWARE_PROFILES_ZH.md`设置可见卡。

## 5. 开跑门槛与停止条件

候选profile显式确认后，先四组一步GPU smoke，核对K6有24条真实主轨迹、λ=0改变token数为零、
mask正确和checkpoint完整。再验证真实保存/恢复；单凭`.pt`文件存在不能证明可恢复。
smoke不是正式结果，正式仍四组×三个种子×200步，不删既有对照来省资源。

若OOM、通信/模型不支持、权限/磁盘不足、资源分配即将结束，就停止并保留日志，不自动降科学规格。
执行配置如需调整，应重新做同源码/同硬件四组smoke，所有正式臂使用一致profile，不跨硬件拼结果。
SwanLab实时记录显式开启，未登录时先停止确认或保留离线记录，不索取聊天里的API Key。

正式命令及验收见`FINAL_CONTROLS_ZH.md`；取得资源与完成部署不等于已经通过这些GPU门槛。
