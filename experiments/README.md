# Experiment protocol

## Methods

| ID | Method | Probe budget | Selection |
|---|---|---:|---|
| `grpo` | standard GRPO | 0 | none |
| `random_b2` | matched-cost control | 2 | stratified random |
| `entropy_b2` | heuristic | 2 | turn entropy |
| `linear_ucb_b2` | ProbeGRPO | 2 | learned value per rollout token |

Use seeds `17`, `42`, and `101`. Tune on seed `17`; freeze all choices before running the other two.

## Required metadata

Every run records:

- ProbeGRPO and verl git revisions, plus the RAGEN environment-source revision when applicable;
- model revision and tokenizer/chat-template hash;
- complete resolved Hydra config;
- CUDA, driver, torch, transformers, vLLM, verl, and PEFT versions;
- seed and environment split;
- main/probe rollout tokens and GPU hours;
- success, reward, steps, invalid actions, entropy, advantage variance, and replay mismatch rate.

## Ablations

Run only after the main pipeline is stable:

- budget: `1`, `2`, `4`;
- lambda: `0.25`, `0.5`, `1.0`;
- token mask: whole assistant turn versus parsed action span;
- scheduler: random, entropy, LinearUCB.

Do not multiply every setting by three seeds. Select the configuration on one seed, then evaluate the
frozen winner and baselines on all three seeds.

## Result policy

Store machine-readable records in ignored `outputs/*.jsonl`. Commit only redacted summaries,
figures, resolved configs, and a manifest with hashes. Negative and crashed runs stay in the table.
