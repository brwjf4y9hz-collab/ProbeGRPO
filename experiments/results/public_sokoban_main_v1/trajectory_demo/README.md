# 真实轨迹案例：正、负、零 credit

来源为 v1 真实模型实验；按 step 和文件名确定性选取各符号的第一个有效案例，不代表频率或性能。
棋盘只做确定性重放和单动作转移；两条完整 probe suffix 未保存，不能展示或声称重新验证其整条轨迹。
保存的 probe 回报与原主轨迹最终回报不一定相同，因为 factual suffix 也重新采样。

## positive：ΔReward=+1

来源：`rollouts/episodes/step-2/5e76e9e4766ff1a6da6ed84268fae1489d2d3703d0f790af7b3b8f4bfe823973.json`；SHA-256：`16e42d8b4e49461fd66a2083f769b713ac72abd4d73efdc24bf0e4b44306fe59`。

scheduler=`linear_ucb_warmup`，anchor turn（从 0 起）=1；动作 `right` → `left`。

anchor 前棋盘（已校验状态哈希与观察一致）：

```text
######
#    #
# @$.#
##   #
######
######
```

原动作后棋盘：

```text
######
#    #
#  @*#
##   #
######
######
```

替代动作后棋盘：

```text
######
#    #
#@ $.#
##   #
######
######
```

保存的 factual / counterfactual suffix 回报：1.0 / 0.0；额外生成 20 tokens。

仅 assistant response 索引 `[61, 62]` 可接收 credit，环境 observation 不可接收。

## negative：ΔReward=-1

来源：`rollouts/episodes/step-1/47e6377a25612bc759522715990e3fc31c041724e118d0dfb212becb6c6b3f46.json`；SHA-256：`7cb9e5a20750b2b754bd82d0f0873a586ca86e7cc5c9508f499d0a1555e3a2df`。

scheduler=`linear_ucb_warmup`，anchor turn（从 0 起）=1；动作 `up` → `down`。

anchor 前棋盘（已校验状态哈希与观察一致）：

```text
######
###  #
### @#
####$#
### .#
######
```

原动作后棋盘：

```text
######
### @#
###  #
####$#
### .#
######
```

替代动作后棋盘：

```text
######
###  #
###  #
####@#
### *#
######
```

保存的 factual / counterfactual suffix 回报：0.0 / 1.0；额外生成 20 tokens。

仅 assistant response 索引 `[60, 61]` 可接收 credit，环境 observation 不可接收。

## zero：ΔReward=+0

来源：`rollouts/episodes/step-1/3d878b9719ad20a747e9b895ec7468b95cd1026d599c6ff0348d6d4343513aa9.json`；SHA-256：`073be8adce24eec9a4710e4bf956b85912cb8261f7ff0304b92c819b722b6ebf`。

scheduler=`linear_ucb_warmup`，anchor turn（从 0 起）=1；动作 `down` → `right`。

anchor 前棋盘（已校验状态哈希与观察一致）：

```text
######
#@ ###
# $ ##
## ###
# . ##
######
```

原动作后棋盘：

```text
######
#  ###
#@$ ##
## ###
# . ##
######
```

替代动作后棋盘：

```text
######
# @###
# $ ##
## ###
# . ##
######
```

保存的 factual / counterfactual suffix 回报：0.0 / 0.0；额外生成 40 tokens。

仅 assistant response 索引 `[56, 57]` 可接收 credit，环境 observation 不可接收。

λ=0.5 且二值 Δ∈{-1,0,1} 时，这三种情况的理论局部 advantage 增量分别为 +0.5、-0.5、0。
这是按公式计算的说明，不是从原始 actor 梯度中恢复的数值；完整 GRPO advantage/实际 loss 不在本案例中伪造。

复核命令：`PYTHONPATH=src python scripts/export_trajectory_demo.py <原始 run 目录> --output-dir <新目录>`。
