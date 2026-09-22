# Public Sokoban main-v1 result

This directory is the compact, committed record of the first complete ProbeGRPO comparison. Raw
episode sidecars, trainer logs, and checkpoints remain under ignored AutoDL `outputs/` directories;
the per-seed CSV transcribes their generated `summary.json` files without selecting favorable runs.
Because reward is binary and the evaluation set has 128 items, `success_count` is the exact
numerator implied by each recorded validation mean.

## Frozen protocol

- Model: `Qwen/Qwen3.5-2B`, LoRA rank 32.
- Trainer: current verl at `cf14ded3a448107e70a206fd201817cc1cbae348`.
- Environment source: `ZihanWang314/ragen-datasets` at
  `e2060cf7c51da891a68e0e7437309e5ec052e45b`.
- Split: 512 unique training boards and 128 unique test boards with zero board-layout overlap.
- Exact BFS oracle: every selected board is solvable within the 12-action horizon.
- Training: 50 updates, four prompts per update, four main rollouts per prompt.
- Seeds: 17, 42, and 101.
- Hardware: one 48 GB RTX 4090.
- Probe methods: budget 2 and local credit coefficient 0.5.

The result can be regenerated from the committed CSV without torch, verl, or plotting packages:

```bash
make results
```

To rebuild the CSV from raw AutoDL summaries instead, run:

```bash
python3 scripts/aggregate_sokoban_seeds.py \
  --output-dir experiments/results/public_sokoban_main_v1 \
  outputs/ablation-bbba7a8-mem25-seed17 \
  outputs/ablation-main-v1-seed42 \
  outputs/ablation-main-v1-seed101
```

## Result

See [`summary.md`](summary.md) for the generated table and
[`aggregate.json`](aggregate.json) for machine-readable statistics. Values after `+/-` are sample
standard deviations across the three seeds, not confidence intervals.

The most defensible comparison is LinearUCB-B2 against GRPO:

- success: `31.0% +/- 1.2%` versus `24.0% +/- 2.0%`;
- paired improvement: `+7.0 +/- 0.8` percentage points;
- successful test episodes: LinearUCB exceeds matched-seed GRPO by 9, 10, and 8 out of 128;
- extra rollout tokens: `51.7%` on average, or `1.517x` the main-rollout token count;
- valid probes: `1,138 / 1,200` (`94.8%`).

Surprisal-B2 has the numerically highest mean success (`31.2%`) but costs more extra rollout tokens
(`57.3%`) and has higher cross-seed variability (`3.4` points). The 0.2-point mean difference from
LinearUCB is much smaller than the observed seed variation and is not treated as a win.

## Evidence boundaries

- Three seeds support a portfolio-level reproducibility claim, not a formal significance claim.
- LinearUCB is the most stable scheduler in this run, but it does **not** win the diagnostic
  high-impact-anchor-per-1,000-token metric. Random-B2 averages `11.93`, LinearUCB `11.32`, and
  Surprisal `10.44`; therefore the planned anchor-efficiency claim is not supported.
- LinearUCB misses the original `<=1.5x` rollout-cost target by 1.7 percentage points. Report the
  measured `1.517x` cost rather than rounding it down.
- Budget 2 currently probes the first two sampled episodes within each prompt group, then schedules
  one turn inside each selected episode. It is not yet group-wide top-B selection.
- Seed 17's GRPO arm completed before the PPO mini-batch padding fix and was retained because its
  synthetic rows had zero loss masks. Seeds 42 and 101 use the final configuration throughout. A
  strict configuration audit should rerun seed 17 GRPO before a paper-level claim.

## Source locations on the experiment host

```text
/root/autodl-tmp/ProbeGRPO/outputs/ablation-bbba7a8-mem25-seed17
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed42
/root/autodl-tmp/ProbeGRPO/outputs/ablation-main-v1-seed101
```

The failed padding run remains preserved as `random_b2.failed-padding` and is excluded by the
summarizer because it never reached its declared final step.
