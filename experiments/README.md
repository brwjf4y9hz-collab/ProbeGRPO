# Experiment protocol

## Methods

| ID | Method | Probe budget | Selection |
|---|---|---:|---|
| `grpo` | standard GRPO | 0 | none |
| `random_b2` | matched-cost control | 2 | stratified random |
| `surprisal_b2` | uncertainty heuristic | 2 | chosen-token surprisal |
| `linear_ucb_b2` | ProbeGRPO | 2 | learned value per rollout token |

Use seeds `17`, `42`, and `101`. Tune on seed `17`; freeze all choices before running the other two.

All four rows are runnable on the pinned current-verl stack. Formal runs use the checksum-verified
public RAGEN Sokoban release, a deterministic 512/128 board-deduplicated split with zero layout
overlap, and a 12-turn horizon verified against an exact shortest-path oracle. Tiny handcrafted
levels remain wiring fixtures and never enter result tables.

The backend exposes sampled action-token log probabilities but not the full next-token
distribution. Therefore `surprisal_b2` is named and logged as chosen-token surprisal rather than
being mislabeled as entropy. LinearUCB uses that rollout-available uncertainty statistic. Budget
`B` currently probes the first `B` of four episodes in each prompt group and performs turn
selection within each episode; group-wide top-B selection is future work.

## Required metadata

Every run records:

- ProbeGRPO and verl git revisions, plus the RAGEN environment-source revision when applicable;
- model revision and tokenizer/chat-template hash;
- complete resolved Hydra config;
- CUDA, driver, torch, transformers, vLLM, verl, and PEFT versions;
- seed and environment split;
- main/probe rollout tokens and GPU hours;
- success, reward, steps, invalid actions, chosen-token surprisal, advantage statistics, and
  replay mismatch rate.

## Ablations

Run only after the main pipeline is stable:

- budget: `1`, `2`, `4`;
- lambda: `0.25`, `0.5`, `1.0`;
- token mask: whole assistant turn versus parsed action span;
- scheduler: random, chosen-token surprisal, LinearUCB.

Do not multiply every setting by three seeds. Select the configuration on one seed, then evaluate the
frozen winner and baselines on all three seeds.

Run one frozen seed at a time:

```bash
bash scripts/run_sokoban_ablation.sh /root/autodl-tmp/probegrpo-runtime/verl main
```

Set `SEED` to 17, 42, or 101. The runner refuses reused output directories, records code/data
revisions, and creates `summary.json` plus `summary.md` after all arms complete. The main-v1 run is
complete; its compact record is in [`results/public_sokoban_main_v1`](results/public_sokoban_main_v1).

Regenerate its aggregate table and vector figure with:

```bash
make results
```

Operational failures and their fixes are retained in dated records. See
[`environment/2026-09-21-public-ablation-incidents.md`](environment/2026-09-21-public-ablation-incidents.md)
for the first public-data run, including network, CUDA memory, colocated initialization, and
synthetic-padding incidents.

## Result policy

Store machine-readable records in ignored `outputs/*.jsonl`. Commit only redacted summaries,
figures, resolved configs, and a manifest with hashes. Negative and crashed runs stay in the table.
