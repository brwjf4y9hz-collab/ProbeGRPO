# Sokoban AgentLoop milestone

## Scope

`SokobanEnv` is a real, deterministic 2D transition environment with three handcrafted one-box
levels. These are **integration fixtures**, not RAGEN's benchmark, procedural levels, or a
generalization study. Two levels are train fixtures; one is held out for later checks.
`TinySokobanEnv` remains the independent micro-test for the probing core.

Formal experiments use the public `ZihanWang314/ragen-datasets` Sokoban parquet files pinned at
revision `e2060cf7c51da891a68e0e7437309e5ec052e45b`. The preparation script verifies both source
checksums, translates the released boards into replayable task IDs, keeps the lowest 512 unique
train layouts and 128 unique test layouts, and excludes every test board duplicated in train.
All selected one-box boards have an exact shortest solution of at most 11 moves, below the fixed
12-turn horizon. The environment uses sparse terminal success reward consistently for every arm.

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
top-two margin are **not** exposed by this adapter's backend contract, so the runnable uncertainty
baseline is explicitly named chosen-token surprisal. LinearUCB uses the same available surprisal
feature and records that provenance; it does not pretend that surprisal is full entropy. Random,
surprisal, and LinearUCB can select one turn inside each probed episode. The trainer hook has been
validated in a real one-update GPU gate and applies credit after standard GRPO advantage calculation.

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
improvement result.

## Paired-suffix debug gate

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

The 2026-09-19 paid GPU gate produced eight episodes and two valid probes, with deltas 1.0 and
0.0 and 304 extra response tokens. It did not apply probe credit to the actor update.
The smoke records at most one probe for session 0 of each training prompt group. An invalid
action, replay/context mismatch, or suffix failure produces zero credit with a skip reason.
The formal adapter counts actual model-generated tokens for both successful and failed branches.
No training-benefit claim is justified until the matched-budget public-data runs are complete.

## Trainer advantage gate

The pinned verl v1 trainer calculates standard GRPO advantages and then writes them back to
TransferQueue as nested tensors. The ProbeGRPO hook runs between those two operations. It joins
each batch row to its hashed episode sidecar using the trajectory ID, checks the full response
mask, packs valid probe credit into `[N,A,T]` and `[N,A]` tensors, and changes only the selected
assistant turn. Missing or mismatched sidecars stop the update instead of assigning credit to an
unrelated episode. The hook supports one TransferQueue span per episode and `budget=0..4`.

After syncing the branch, validate the saved real episodes with the installed runtime without
loading Qwen weights:

```bash
/root/autodl-tmp/probegrpo-runtime/verl/.venv/bin/python \
  scripts/check_saved_probe_hook.py \
  outputs/verl-sokoban-probe-smoke-882ea33-run2/rollouts/episodes/step-1
```

The script verifies exact GRPO fallback for `budget=0` and `lambda=0`, then prints changed token
indices for `lambda=0.5`. Probe deltas are now scaled by `max(1, largest absolute valid delta)`
in the batch, without mean-centering. For `[1.0, 0.0]`, normalized values are `[1.0, 0.0]`: a zero-delta
anchor receives no credit. The first GPU update on 2026-09-20 used the earlier centered z-score
and changed four tokens; it is a wiring gate, not a result comparable with future experiments.

For the one-update training gate, install the idempotent hook in the pinned external checkout and
run with a new output directory:

```bash
export OUTPUT_DIR=/root/autodl-tmp/ProbeGRPO/outputs/verl-sokoban-probe-train-smoke-1
bash scripts/run_verl_sokoban_probe_train_smoke.sh \
  /root/autodl-tmp/probegrpo-runtime/verl
```

The installer checks the exact verl commit and keeps the original trainer in a neighboring
`.py.probegrpo.backup` file. Training logs must show `probe/valid`, `probe/changed_tokens`, and
`probe/credit_abs_sum`; a completed optimizer step alone does not establish that credit was used.

## Public-data matched-budget experiments

The AgentLoop reads `probe.enabled`, `probe.budget`, and `probe.scheduler` before model generation.
`budget=0` triggers no paired suffixes and exactly retains GRPO advantages. With four sampled
episodes per prompt, `budget=B` probes sessions `[0, B)`, and the configured scheduler selects one
turn inside each selected episode.
Setting `lambda_coef=0` retains GRPO advantages but still runs and counts any configured probes;
it is therefore not a zero-cost baseline.
The older `PROBEGRPO_DEBUG_PROBE=1` path remains a rollout-only diagnostic when training probe
config is absent. This is not yet group-wide top-B selection across all four trajectories, and that
limitation must remain explicit in reports.

The formal runner downloads and verifies the public release, writes a run manifest, refuses to mix
an existing output directory, and runs the four main arms sequentially:

```bash
bash scripts/run_sokoban_ablation.sh \
  /root/autodl-tmp/probegrpo-runtime/verl main
```

The first pass uses seed 17. It compares GRPO, Random-B2, Surprisal-B2, and LinearUCB-B2 with the
same model, public split, main rollout count, horizon, and probe budget. The script writes
`summary.json` and `summary.md`. Run the `lambda` and `budget` phases only after the main pass is
stable; freeze the chosen configuration before repeating with seeds 42 and 101.
