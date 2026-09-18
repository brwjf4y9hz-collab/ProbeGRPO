# ProbeGRPO

ProbeGRPO is an internship-oriented Agent-RL project that demonstrates a complete path from
multi-turn environment rollouts to fine-grained policy updates. It adds **budget-aware,
counterfactual turn probing** to GRPO: the system replays selected decision points, tries a different
valid action, measures the change in environment reward, and applies the resulting credit only to
the corresponding assistant turn.

The goal is a credible, inspectable engineering project—not a claim that counterfactual credit
assignment itself is new. The differentiators are a small reusable core, deterministic replay checks,
matched-budget schedulers, current verl integration, RAGEN-derived task adapters, and explicit cost
accounting.

## What is implemented

- A framework-independent `ReplayableEnv` contract with deterministic prefix replay and state hashes.
- Structured `TurnRecord`, `Anchor`, `ProbeResult`, and experiment metrics types.
- Random, entropy, and online LinearUCB anchor schedulers.
- Paired factual/counterfactual suffix probing with strict replay validation.
- Probe-aware GRPO advantage shaping with exact assistant-turn token masks.
- A lazy torch/verl adapter that keeps the core package dependency-free.
- A deterministic tiny Sokoban environment and a one-command CPU smoke test.
- A pinned current-verl bootstrap and five-update `Qwen/Qwen3.5-2B` GRPO gate for one 48 GB GPU.

The GPU scripts are ready, but the actual GPU run has not been executed yet. Sokoban and WebShop
will reuse RAGEN's environment ideas while targeting verl's current `AgentLoop` interface.

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
cd ProbeGRPO
mamba env create -f environment.yml
conda activate probegrpo
make check
```

If the environment already exists, only activate it and run `make check`. This local macOS
environment is for development and CPU tests; create a separate environment on the NVIDIA Linux
host for verl, PyTorch CUDA, and vLLM.

Expected final line:

```text
ProbeGRPO CPU data-flow smoke test passed
```

## Current verl + Qwen3.5

The default model is [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B). RAGEN's released
training dependencies are too old for this model, so ProbeGRPO pins a current verl revision and its
frozen dependency lock. On an AutoDL Linux instance with one 48 GB NVIDIA GPU:

```bash
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
bash scripts/check_verl_stack.sh /path/to/persistent-workspace/probegrpo-runtime/verl
bash scripts/run_verl_grpo_smoke.sh /path/to/persistent-workspace/probegrpo-runtime/verl
```

The first paid run deliberately uses GSM8K rather than an agent environment. It verifies model
loading, vLLM generation, LoRA/FSDP2 updates, grouped GRPO, checkpointing, and resume before we add
Sokoban complexity. See [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

The ProbeGRPO integration adapter expects the trainer batch to contain:

- `advantages`: `[batch, tokens]`
- `probe_turn_masks`: `[batch, anchors, tokens]`
- `probe_deltas`: `[batch, anchors]`
- `probe_valid`: `[batch, anchors]`

It standardizes valid deltas and adds `lambda * delta_z * turn_mask` to the base advantages.

## Project milestones

1. **Core correctness:** `make check` passes; deterministic replay and masked advantages are tested.
2. **GPU stack gate:** Qwen3.5-2B completes five standard GRPO updates and resumes on current verl.
3. **Agent rollout:** a RAGEN-derived Sokoban task completes multi-turn current-verl `AgentLoop`
   rollouts and deterministic prefix replay.
4. **Matched-budget comparison:** GRPO, random probe, entropy probe, and LinearUCB use identical
   main-rollout and probe budgets.
5. **Portfolio result:** publish curves, cost table, replayable WebShop traces, and a short demo.

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

> Built ProbeGRPO, a Qwen3.5-based Agent-RL system on current verl with deterministic trajectory
> replay, budget-aware counterfactual credit probing, turn-level advantage shaping, and matched-cost
> evaluation on Sokoban and WebShop.
