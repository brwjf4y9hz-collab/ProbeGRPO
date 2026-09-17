# ProbeGRPO

ProbeGRPO is an internship-oriented Agent-RL project that demonstrates a complete path from
multi-turn environment rollouts to fine-grained policy updates. It adds **budget-aware,
counterfactual turn probing** to GRPO: the system replays selected decision points, tries a different
valid action, measures the change in environment reward, and applies the resulting credit only to
the corresponding assistant turn.

The goal is a credible, inspectable engineering project—not a claim that counterfactual credit
assignment itself is new. The differentiators are a small reusable core, deterministic replay checks,
matched-budget schedulers, RAGEN/verl integration, and explicit cost accounting.

## What is implemented

- A framework-independent `ReplayableEnv` contract with deterministic prefix replay and state hashes.
- Structured `TurnRecord`, `Anchor`, `ProbeResult`, and experiment metrics types.
- Random, entropy, and online LinearUCB anchor schedulers.
- Paired factual/counterfactual suffix probing with strict replay validation.
- Probe-aware GRPO advantage shaping with exact assistant-turn token masks.
- A torch/RAGEN adapter that keeps the core package dependency-free.
- A deterministic tiny Sokoban environment and a one-command CPU smoke test.
- RAGEN configuration overlays for `Qwen/Qwen3.5-2B` and an optional 4B stretch run.

## Architecture

```text
main K=4 rollouts
       |
       v
 TurnRecord extraction -----> anchor scheduler
                                  |  budget B=2
                                  v
                           deterministic replay
                              /            \
                       factual suffix   counterfactual suffix
                              \            /
                               delta reward
                                    |
                                    v
trajectory GRPO advantage + turn-local probe credit
                                    |
                                    v
                              policy update
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data contract and integration boundary.

## Quick start (CPU, no model download)

The core and smoke test intentionally require only the Python standard library.

```bash
make check
```

Expected final line:

```text
ProbeGRPO smoke test passed
```

## RAGEN + Qwen3.5

The default model is [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B). It is recent,
small enough for prototype LoRA runs, and officially supports vLLM. `Qwen/Qwen3.5-4B` is provided
as a stretch configuration for a 48 GB GPU or two 24 GB GPUs.

RAGEN pins its own verl version, so run the compatibility check before a real training job:

```bash
bash scripts/bootstrap_ragen.sh /path/to/workspace
bash scripts/check_ragen_compat.sh /path/to/workspace/RAGEN
```

Then use the generated overlay with the RAGEN checkout. The integration adapter expects the trainer
batch to contain:

- `advantages`: `[batch, tokens]`
- `probe_turn_masks`: `[batch, anchors, tokens]`
- `probe_deltas`: `[batch, anchors]`
- `probe_valid`: `[batch, anchors]`

It standardizes valid deltas and adds `lambda * delta_z * turn_mask` to the base advantages.

## Project milestones

1. **Core correctness:** `make check` passes; deterministic replay and masked advantages are tested.
2. **Framework smoke:** Qwen3.5-2B completes five RAGEN updates on tiny Sokoban.
3. **Matched-budget comparison:** GRPO, random probe, entropy probe, and LinearUCB use identical
   main-rollout and probe budgets.
4. **Portfolio result:** publish curves, cost table, replayable WebShop traces, and a short demo.

For an internship portfolio, a well-explained negative result is acceptable. Do not fabricate gains;
report when extra probes improve credit diagnostics but fail to improve final reward.

## Evaluation metrics

- task reward and success rate;
- invalid-action rate and mean episode length;
- reward/advantage variance;
- total rollout tokens, GPU hours, and wall-clock time;
- high-impact anchors found per 1,000 extra rollout tokens;
- Sokoban rank correlation against an exhaustive counterfactual oracle.

## Repository policy

- Raw traces, checkpoints, and W&B files belong under ignored `artifacts/`, `outputs/`, or `wandb/`.
- Publish LoRA adapters rather than full model weights.
- Never commit API keys, cookies, WebShop sessions, or unredacted external traces.
- Results must include the exact config, git revision, seeds, and unsuccessful runs.

## Resume bullet

> Built ProbeGRPO, a Qwen3.5-based Agent-RL system on RAGEN/verl with deterministic trajectory
> replay, budget-aware counterfactual credit probing, turn-level advantage shaping, and matched-cost
> evaluation on Sokoban and WebShop.

