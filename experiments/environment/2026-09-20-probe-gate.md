# Sokoban probe trainer gate — user-reported console evidence

Source: AutoDL terminal output pasted by the operator on 2026-09-20. Raw logs and episode
sidecars remain under `/root/autodl-tmp/ProbeGRPO/outputs/`. This is a transcription, not an
independently rerun experiment.

## One-update result at ProbeGRPO `076136b`

- The first launch failed during vLLM startup before any rollout because `OMP_NUM_THREADS`
  resolved to a non-positive value. Retrying with `OMP_NUM_THREADS=1` in the shell and Ray
  runtime environment completed one actor update.
- Output directory: `outputs/verl-sokoban-probe-train-076136b-omp1/`.
- `training/global_step=1`, `probe/valid=2`, `probe/skipped=6`,
  `probe/extra_tokens=875`, `probe/changed_tokens=4`, `probe/credit_abs_sum=2.0`.
- Eight saved episodes were matched to eight rollout rows. Dense probe shapes were
  `N=8, A=1, T=462`. The saved-sidecar checker confirmed that `budget=0` and `lambda=0`
  leave GRPO advantages unchanged.
- The two valid probes had raw deltas `1.0` and `0.0`. The then-current centered z-score
  mapped them to `+1` and `-1`, changing two assistant tokens in each episode. The negative
  value on the zero-delta probe was relative to the batch, not evidence that its factual
  action was worse.

## Interpretation and next gate

This proves that a real model-generated paired probe reached the verl v1 advantage hook and
actor update. It is **not** a performance comparison or an ablation. After this run, the
normalization was changed to zero-preserving `max(1, largest absolute delta)` scaling: `[1.0, 0.0]` now maps to
`[1.0, 0.0]`. That revised calculation requires saved-sidecar and GPU revalidation before
any matched-budget experiments. Do not mix the first run's metrics with later experiments.
