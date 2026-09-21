# Public Sokoban ablation run log — 2026-09-21

## Reproducibility scope

- ProbeGRPO branch: `codex/verl-gpu-bootstrap`
- Model: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`
- verl revision: `cf14ded3a448107e70a206fd201817cc1cbae348`
- Public dataset: `ZihanWang314/ragen-datasets`, revision
  `e2060cf7c51da891a68e0e7437309e5ec052e45b`
- Derived split: 512 unique train boards and 128 unique test boards, with test layouts
  overlapping train removed.
- Horizon: 12 moves. Exact BFS verification found every selected board solvable within the
  horizon; maximum shortest solution length is 11 for train and 9 for test.
- Main-rollout shape: four prompts per update times four sampled trajectories = 16 trajectories.

Source parquet checksums and derived parquet checksums are written to the generated data
`manifest.json`. Every training arm writes `run_manifest.json`, resolved trainer logs, standard
rollout JSONL, and one full episode sidecar per trajectory under its ignored `outputs/` directory.
Raw outputs are intentionally not committed because they can be large; result summaries and this
failure log are committed.

## Observed pre-training evaluation

The memory-fixed run completed all initialization and evaluated the 128-example public test set:

- initial success/reward mean: `0.2578125` (33/128);
- validation episode turns: min `2`, max `24`, mean `18.515625` in verl's assistant/user-turn
  counting convention.

This is a pretrained-model baseline before any optimizer update, not a ProbeGRPO result.

## Incident 1 — public dataset download unavailable

**Symptom:** `Network is unreachable` followed by a closed Hugging Face HTTP client.

**Cause:** AutoDL no-card mode had no active network acceleration route.

**Resolution:** source `/etc/network_turbo`, disable offline mode only during preparation, disable
Xet, and use a 600-second download timeout. The preparation step verifies immutable SHA-256
checksums before producing data.

## Incident 2 — actor entropy CUDA OOM

**Symptom:** `entropy_from_logits` attempted one additional 11.97 GiB allocation with only
7.17 GiB free.

**Cause:** full-vocabulary entropy was materialized for too many tokens at once.

**Resolution (commit `f9f27bd`):** enable verl's chunked entropy computation with chunk size 256,
set actor dynamic token budget to 2048, cap response length at 1024, and reduce PPO mini-batch size.
All experiment arms use the same corrected configuration.

## Incident 3 — colocated initialization worker killed

**Symptom:** the FSDP actor worker exited during initial vLLM weight synchronization without a
Python exception; Ray reported an unexpected EOF.

**Cause:** the colocated vLLM server reserved 45% of the 48 GB GPU, leaving insufficient transient
headroom while FSDP synchronized weights.

**Resolution (commit `bbba7a8`):** reduce vLLM `gpu_memory_utilization` to `0.25`. The next run
completed actor/vLLM initialization and initial validation.

## Incident 4 — synthetic padding had no episode sidecar

**Symptom:** Random-B2 upsampled 16 real trajectories to 32 rows and then failed with
`FileNotFoundError` for a synthetic row's probe sidecar.

**Cause:** `ppo_mini_batch_size=8` made the v1 trainer require a multiple of 32. Synthetic padding
rows do not correspond to real environment episodes and therefore correctly have no replay trace.

**Resolution (commit `31e0da9`):** set `ppo_mini_batch_size=4`, making the required multiple equal
the 16 real trajectories. This avoids synthetic training rows rather than fabricating probe credit.
The ablation launcher also skips an arm whose final step sidecars already exist, allowing the
completed GRPO arm to be retained while failed probe arms are rerun.

## Evidence boundaries

- The GRPO arm reached completion before the first Random-B2 sidecar failure, but its final metric
  values must be read from its saved log and generated summary; they are not inferred here.
- The failed Random-B2 directory is preserved with a `.failed-padding` suffix.
- No improvement claim is allowed until all four matched-budget arms complete and the generated
  `summary.json`/`summary.md` are reviewed.
