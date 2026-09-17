# GPU runbook

## Read this first

The inspected RAGEN revision (`d97bb328...`) pins `vllm==0.8.2`. Qwen3.5 requires a much newer
Transformers/vLLM stack, so installing RAGEN's default dependencies and merely changing the model
name is not valid. The repository includes a compatibility checker precisely to prevent that silent
failure.

The CPU core is already runnable. GPU validation is a separate milestone and must happen on the
actual NVIDIA host.

## Recommended hardware

- Default: one 48 GB GPU or two 24 GB GPUs for `Qwen/Qwen3.5-2B` LoRA.
- Development fallback: reduce environment groups to 2 and group size to 2.
- Stretch: `Qwen/Qwen3.5-4B` on two GPUs only after the 2B run is stable.

## Safe bring-up order

1. Create a Python 3.12 environment.
2. Run `scripts/bootstrap_ragen.sh` in a disposable workspace.
3. Run `scripts/check_ragen_compat.sh`; do not ignore its old-vLLM warning.
4. Build a Qwen3.5-capable verl/vLLM environment, then install RAGEN with `--no-deps`.
5. Verify plain inference and chat templating for Qwen3.5 before starting RL.
6. Run standard GRPO for five Sokoban updates with probes disabled.
7. Run five updates for each scheduler with `budget=1`.
8. Inspect token masks and replay-state mismatch rate.
9. Only then raise the budget to 2 and start WebShop setup.

Do not pin speculative package versions in a public README before the GPU host verifies them. Record
the successful lockfile or container digest under `experiments/environment/` after validation.

## Five-update acceptance checklist

- no NaN/Inf in loss, reward, entropy, log-probabilities, or probe deltas;
- exactly four main trajectories per group;
- probe budget never exceeds configured `B`;
- replay-state mismatch rate below 1% on deterministic Sokoban;
- `budget=0` advantages match standard GRPO within floating-point tolerance;
- total extra rollout tokens and wall-clock time are logged;
- a checkpoint can resume for at least one additional update.

## Qwen3.5-specific checks

- Confirm the installed Transformers version recognizes `Qwen3_5ForConditionalGeneration`.
- Confirm vLLM can serve the model and return token log probabilities.
- Keep text-only observations for Sokoban/WebShop even though Qwen3.5 is multimodal.
- Save the exact chat template; turn masks depend on its assistant delimiters.
- Validate LoRA target discovery before using `all-linear`, especially around Gated DeltaNet layers.

