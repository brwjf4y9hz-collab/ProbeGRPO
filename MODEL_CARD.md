# ProbeGRPO adapter model card

This file becomes the release card for LoRA adapters trained by ProbeGRPO.

## Base model

- Default: `Qwen/Qwen3.5-2B`
- Stretch: `Qwen/Qwen3.5-4B`

## Training

Fill in the exact base-model revision, data/environment splits, seeds, number of updates, total
rollout tokens, hardware, resolved configuration, and ProbeGRPO commit.

## Intended use

Research and portfolio demonstrations in sandboxed Sokoban and WebShop environments.

## Limitations

The model is not intended for real purchases or unsandboxed web actions. Environment reward can be
incomplete and may encourage shortcuts. Counterfactual estimates are local, noisy, and restricted to
enumerated legal actions.

## Evaluation

Report every baseline using the same main-rollout and probe budget. Include all seeds and failed
runs; do not select the best checkpoint using the test set.

