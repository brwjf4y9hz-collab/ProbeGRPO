# Sokoban AgentLoop milestone

## Scope

`SokobanEnv` is a real, deterministic 2D transition environment with three handcrafted one-box
levels. These are **integration fixtures**, not RAGEN's benchmark, procedural levels, or a
generalization study. Two levels are train fixtures; one is held out for later checks.
`TinySokobanEnv` remains the independent micro-test for the probing core.

The episode driver is dependency-free. The optional `SokobanAgentLoop` subclasses the pinned
verl `AgentLoopBase`, uses its Continuous Token methods, calls its model server once per action,
and returns `AgentLoopOutput` with environment-derived `reward_score`. No LLM judge is involved.
Real model generation and actor training were validated in a one-update AutoDL smoke on
2026-09-19; CPU tests use scripted actions and are not learning results.

## Data flow

```text
task_id + env_seed -> reset -> board observation
     -> Qwen tokens -> parse one action -> env.step -> next board -> ...
     -> Episode (tokens, masks, before/after hashes, action prefixes, reward)
     -> AgentLoopOutput -> verl TransferQueue -> standard GRPO update
```

Response coordinates include assistant tokens AND intermediate environment feedback. Only the
assistant positions receive mask=1. Initial prompt positions are outside these coordinates. The
final response ends at an assistant token. A context merge that alters a previously recorded
assistant token fails instead of silently misaligning future probe credit.

Generated malformed text and blocked moves consume an action attempt and leave the board unchanged;
the step count changes, so the full-state hash changes. Episodes stop on success, environment
horizon, token budget, or empty generation. No action is executed after its tokens were truncated.

Sampled token logprobs are available when the backend supplies them. Full-distribution entropy and
top-two margin are **not** exposed by this adapter's backend contract. Episode JSON records those
as null. `Episode.to_trace(...)` requires explicit actor statistics aligned to the response axis;
the CPU demo supplies labelled synthetic statistics only. Do not feed placeholder zeros into the
entropy/LinearUCB scheduler. An optional random-anchor debug gate now generates paired
factual/counterfactual suffixes with the same sampling seed. It records the result in the episode
sidecar but **does not apply credit to GRPO advantages**. Group-wide scheduling, measured
entropy/LinearUCB and the trainer advantage hook remain future work.

## Run in order

Local CPU test (no torch, no downloads):

```bash
make check
make agent-smoke
```

After syncing this code and reinstalling editable ProbeGRPO in the existing remote runtime, run
the actual Qwen/verl tokenization check first. It loads the cached tokenizer only, uses scripted
actions and exercises the real `AgentLoopOutput.as_dict()` conversion. On AutoDL's 2 GB no-card
mode even runtime imports may exceed RAM; use a sufficiently sized CPU or GPU instance.

```bash
export HF_HOME=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface
export OMP_NUM_THREADS=1
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/check_sokoban_tokenizer.py
```

Only after that check passes, one real model/GRPO update:

```bash
bash scripts/run_verl_sokoban_smoke.sh /root/autodl-tmp/probegrpo-runtime/verl
```

Default output: `outputs/verl-sokoban-smoke/`. Standard verl JSONL lacks custom trace fields, so
the adapter also writes one sidecar per episode under `rollouts/episodes/step-N/`. It retains the
complete response mask, action prefixes, observations and full-state hashes. Files use hashed IDs
and exclusive creation; choose a new OUTPUT_DIR for a rerun with reused identifiers.

```bash
PYTHONPATH=src python3 scripts/inspect_sokoban_episodes.py \
  outputs/verl-sokoban-smoke/rollouts/episodes/step-1
```

Acceptance: 2 prompts x 4 sampled episodes = 8 trajectories; sidecar masks cover exactly the
assistant tokens, all action-prefix replays match, and bounded episodes return environmental
rewards. Solving a fixture is useful but not guaranteed by a pretrained model. Equal rewards can
still yield zero GRPO advantages. The 2026-09-19 remote smoke completed one real Qwen/verl GRPO
update, wrote 8 sidecars and passed the replay/mask inspector for all 8. One of eight fixture
episodes succeeded. This is an integration check, not a held-out benchmark or a ProbeGRPO
improvement result. The paid GPU paired-probe gate has not yet been run.

## Paired-suffix debug gate (next paid-GPU check)

First sync the local branch to AutoDL. Before loading model weights, run the scripted tokenizer
check with the probe switch enabled. This exercises the same pair orchestration but uses scripted
`up` actions, so it is not a real-model result:

```bash
export HF_HOME=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
PROBEGRPO_DEBUG_PROBE=1 \
  /root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python scripts/check_sokoban_tokenizer.py
```

Once this passes, run one real update with a **new output directory**. `MODEL_PATH` should point
to the already downloaded snapshot; the exact revision below was verified on the 2026-09-19 host:

```bash
export MODEL_PATH=/root/autodl-tmp/probegrpo-runtime/verl/data/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
bash scripts/run_verl_sokoban_probe_smoke.sh /root/autodl-tmp/probegrpo-runtime/verl
python3 scripts/inspect_sokoban_probes.py \
  outputs/verl-sokoban-probe-smoke/rollouts/episodes/step-1
```

The smoke records at most one probe for session 0 of each training prompt group. An invalid
action, replay/context mismatch, or suffix failure produces zero credit with a skip reason.
The inspector reports attempted/valid/skipped probes and valid-probe token cost. Failed attempts
currently report zero tokens even if generation began, so they cannot yet support a rigorous
total-cost claim. No training-benefit claim is justified until the trainer advantage hook and
matched-budget baselines are tested.
