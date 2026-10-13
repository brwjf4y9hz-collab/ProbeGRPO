# 证据与实现来源

引用的报告是当时记录；当前代码用于说明实现入口。归档/测试数字只描述原记录，不说明本轮重新执行。SHA-256核对文件内容，不能证明历史事件真实性。包里不引用本地 artifacts 路径作为必需交付文件。

| 相对项目根目录的文件 | 作用 | 关联问题 | SHA-256 |
|---|---|---|---|
| [docs/results/README.md](../../docs/results/README.md) | 已有实验/核验/分析记录 | P07 | `56fc4a67c4b04a9d254ce4ab34e3716161ad643d2b974c544135423242e14787` |
| [docs/results/figure-provenance.json](../../docs/results/figure-provenance.json) | 已有实验/核验/分析记录 | 结果图补充 | `de437149dcf03af23a79ec290aa6061cc169c54c1d1c92d8eb18836e1b712cc2` |
| [docs/results/figures/01_probegrpo.png](../../docs/results/figures/01_probegrpo.png) | 已有实验/核验/分析记录 | 结果图补充 | `fe03969fa7f04f45fb983356b426e41210163e82d832521be7c9ad77f7d59d4f` |
| [docs/results/figures/01_probegrpo.svg](../../docs/results/figures/01_probegrpo.svg) | 已有实验/核验/分析记录 | 结果图补充 | `654076713963c4cb19612062170fc12971435cf5cc61107a34d44138a65f477c` |
| [docs/results/plot_results.py](../../docs/results/plot_results.py) | 已有实验/核验/分析记录 | 结果图补充 | `581e597a679ffd9659088c1d6a7623ea87043ff851a07e394fb6bbf0380a366b` |
| [docs/results/requirements-viz.txt](../../docs/results/requirements-viz.txt) | 已有实验/核验/分析记录 | 结果图补充 | `b948456519975d7b78938a07ffdefa26ac78f892616e9102226ad4cd0e8fc34e` |
| [experiments/environment/2026-09-19-gpu-gate.md](../../experiments/environment/2026-09-19-gpu-gate.md) | 已有实验/核验/分析记录 | P01 | `0666d5fcbd606850a02884bbf35857fa916f3de148b7d9335f697401e8afb82a` |
| [experiments/environment/2026-09-20-probe-gate.md](../../experiments/environment/2026-09-20-probe-gate.md) | 已有实验/核验/分析记录 | P02, P03 | `019aa9064abb1a8a98fc865d88d1975fce64e804a7a2a3254bf60a1f09e592e2` |
| [experiments/environment/2026-09-21-public-ablation-incidents.md](../../experiments/environment/2026-09-21-public-ablation-incidents.md) | 已有实验/核验/分析记录 | P04, P05, P06 | `09db16205ca18896f18faf1a4254dcc8a329e70d4e9f8488ceed2c686477f0a3` |
| [experiments/results/public_sokoban_main_v1/README.md](../../experiments/results/public_sokoban_main_v1/README.md) | 已有实验/核验/分析记录 | P03, P05, P07 | `21ad1d6b8c5cf80263da0912a07de4add621f9b97b4d2bd3de103334da7e8898` |
| [experiments/results/public_sokoban_main_v1/aggregate.json](../../experiments/results/public_sokoban_main_v1/aggregate.json) | 已有实验/核验/分析记录 | P07 | `81162de5476434273ea31dbf1220e2998292aac80aab48cdcb838da7860a63f8` |
| [repro/ARTIFACTS.md](../../repro/ARTIFACTS.md) | 已有实验/核验/分析记录 | P08 | `30cfdca2626f1f28ea43530f5f0e7ff49990b9364b5cac2cac9027672ca6a207` |
| [repro/validation/README.md](../../repro/validation/README.md) | 已有实验/核验/分析记录 | P06, P08 | `ab8bf294cb9381b8b3d4e98b175d7947d90de0a338d392538e37949b15695b16` |
| [scripts/artifact_manifest.py](../../scripts/artifact_manifest.py) | 当前实现/检查入口 | P08 | `ac26fb95568cdd157c466a70c0e1eebff480f789bdb426bf95349b5d58a026bd` |
| [scripts/bootstrap_verl.sh](../../scripts/bootstrap_verl.sh) | 当前实现/检查入口 | P01 | `30cd5481c2787024013276eb8066709156b69da70c2ad2c5aa9d61d1203b276d` |
| [scripts/prepare_public_sokoban_data.py](../../scripts/prepare_public_sokoban_data.py) | 当前实现/检查入口 | P06 | `81b1e7b11bb42d96ebfbeb09eb4aac3e354dde47ce5969d1057bc8c0e704da54` |
| [scripts/reproduce_results.py](../../scripts/reproduce_results.py) | 当前实现/检查入口 | P08 | `fc1f035eb6acf69dd0b8f237276e60b62999b836d2af94a24bd4de47dfb4acac` |
| [scripts/run_public_sokoban_train.sh](../../scripts/run_public_sokoban_train.sh) | 当前实现/检查入口 | P04, P05 | `0fbe626c412dab8ae0fa322e1123632865db7511eba02e67e9d86c0c69c6ac94` |
| [scripts/run_verl_grpo_smoke.sh](../../scripts/run_verl_grpo_smoke.sh) | 当前实现/检查入口 | P01 | `b5bf90100bfe8bc12f24c2318d51a74d5d270a431b43eea306b9ace9547df541` |
| [scripts/run_verl_sokoban_smoke.sh](../../scripts/run_verl_sokoban_smoke.sh) | 当前实现/检查入口 | P02 | `26ca303b2144b7785086685ca8218ba6e4130a50d08a6b71f9eeb30fe481d67a` |
| [scripts/summarize_sokoban_ablation.py](../../scripts/summarize_sokoban_ablation.py) | 当前实现/检查入口 | P05 | `b63b0ea3978ceb4ef56b56074bbdb800bc35a36c1fd2f3ba7c0ba8030259f918` |
| [src/probegrpo/advantage.py](../../src/probegrpo/advantage.py) | 当前实现/检查入口 | P03 | `b0904d8b9ee38ba3d6b14d7a00e45a945e97ae4205e9fe130b99c5a17db484b5` |
| [tests/test_ablation_summary.py](../../tests/test_ablation_summary.py) | 当前实现/检查入口 | P05 | `45f26174367cc5bd66d17a1f6b3ce2af6ab78e73fa73566e7f674d00ad51550c` |
| [tests/test_pipeline.py](../../tests/test_pipeline.py) | 当前实现/检查入口 | P03 | `7b8dca28dc42ed848306712e383a7784debcbe85a54914d22ba1c6a3715b5220` |
| [tests/test_public_sokoban.py](../../tests/test_public_sokoban.py) | 当前实现/检查入口 | P06 | `13af9e931b8715370289b797374a13ce4c2f6697034ecc82c1f47d3ac27b25c1` |
| [tests/test_verl_v1_sidecar.py](../../tests/test_verl_v1_sidecar.py) | 当前实现/检查入口 | P03 | `00fbc290d79f529ba8fee494b2cd71052e50f063250f3f681f46804cff296350` |

证据选择：只打包台账直接引用的既有仓库文件、复盘文字和核验元数据；不递归复制运行输出、不重新上传隐藏评分材料。原报告的历史下一步可能已在后续版本完成，以本台账状态和当前结果分析为准。
