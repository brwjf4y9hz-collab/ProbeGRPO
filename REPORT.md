# ProbeGRPO technical report

> Status: complete three-seed public-Sokoban portfolio result; WebShop is an optional extension.

## Abstract

Long-horizon language-model agents normally receive a terminal reward for an entire trajectory.
GRPO can compare sampled trajectories for the same prompt, but every generated action token in one
trajectory still receives the same trajectory-level signal. ProbeGRPO adds a bounded amount of
environment interaction: it selects a small number of assistant turns, deterministically replays
the action prefix, substitutes a legal alternative action, and converts the factual-minus-
counterfactual terminal-reward difference into credit for only the selected turn. The system is
implemented on current verl with a Qwen3.5-2B LoRA actor and a replayable Sokoban AgentLoop.

On a checksum-pinned public RAGEN Sokoban split, four methods were trained for 50 updates using
three random seeds. LinearUCB-B2 increased held-out success from `24.0% +/- 2.0%` for GRPO to
`31.0% +/- 1.2%`, a paired improvement of `7.0 +/- 0.8` percentage points, while adding `51.7%`
rollout tokens. Surprisal-B2 reached a similar mean at higher cost; Random-B2 was less stable. These
results establish a reproducible engineering result, not statistical significance or a claim that
counterfactual credit assignment is new.

## 1. Problem and design objective

For an episode with assistant turns `a_1 ... a_T` and final reward `R`, trajectory-level GRPO assigns
one relative trajectory advantage to every response token. This is inexpensive but coarse: a
correct early action in a failed episode is penalized with later mistakes, while irrelevant actions
in a successful episode receive positive credit.

ProbeGRPO asks a narrower engineering question: under a fixed additional rollout budget, can real
environment replay provide useful turn-local credit without an LLM judge? The design requirements
are:

1. factual and counterfactual suffixes start from an identical verified state;
2. extra rollout cost is explicitly measured in generated tokens;
3. probe credit changes only the selected assistant turn;
4. budget zero and credit coefficient zero exactly recover ordinary GRPO;
5. scheduler comparisons use identical main-rollout, dataset, model, and training budgets.

## 2. System

### 2.1 Main AgentLoop

Each update samples four prompts and four trajectories per prompt. The Qwen actor emits one legal
action string per assistant turn; Sokoban returns the next board as an environment observation.
Each recorded turn includes the action prefix before the current action, state hash, legal actions,
sampled action-token log probabilities, response-token indices, validity, and final trajectory
reward. Environment observations occupy response positions but have an assistant mask of zero.

### 2.2 Counterfactual probe

For an anchor turn `t`, two fresh environments reset to the same task ID and seed, then replay
`a_1 ... a_(t-1)`. The probe is discarded if either reconstructed observation or state hash differs
from the recorded anchor. The factual branch executes `a_t`; the counterfactual branch executes a
different legal action. Both then generate suffixes under the same sampling seed. Credit is

```text
delta_t = factual_terminal_reward - counterfactual_terminal_reward
```

A positive delta means the original action was better than the tested alternative; a negative
delta means the alternative was better. Zero delta is valid evidence but produces no update.

### 2.3 Advantage integration

The sidecar converts sparse probe records to dense tensors:

```text
probe_turn_masks [N, A, T]
probe_deltas     [N, A]
probe_valid      [N, A]
```

After verl computes standard GRPO advantages and before the actor update, ProbeGRPO applies

```text
A_final = A_GRPO + lambda * normalize(delta) * turn_mask
```

with `lambda=0.5`. The mask covers only the selected assistant action tokens, never prompts,
environment observations, padding, or other turns. The integration is tested for exact fallback
when budget is zero, lambda is zero, no probe is valid, or all deltas are zero.

### 2.4 Schedulers

- **Random-B2:** position-stratified random turn selection.
- **Surprisal-B2:** prioritizes sampled actions with high chosen-token surprisal. It is not called
  entropy because the rollout backend does not expose the full next-token distribution.
- **LinearUCB-B2:** an online linear contextual bandit estimating absolute probe delta per extra
  rollout token with an uncertainty bonus and random warm-up.

Budget two currently chooses the first two of four sampled episodes in a prompt group and then one
turn within each chosen episode. Group-wide top-B episode-and-turn selection remains future work.

## 3. Experimental setup

| Component | Setting |
|---|---|
| Policy | `Qwen/Qwen3.5-2B` |
| Adaptation | BF16 LoRA, rank 32, alpha 64 |
| Trainer | current verl, FSDP2 actor and vLLM rollout |
| Hardware | one RTX 4090 48 GB |
| Dataset | pinned public RAGEN Sokoban release |
| Split | 512 train / 128 held-out test boards |
| Leakage control | board-layout deduplication and zero train/test overlap |
| Horizon | 12 actions; every selected board verified solvable by exact BFS |
| Training | 50 updates, 16 main trajectories per update |
| Probe | budget 2, lambda 0.5 |
| Seeds | 17, 42, 101 |

The exact model, verl, and dataset revisions are stored in
`experiments/results/public_sokoban_main_v1/metadata.json`. All four methods use the same public
split and main rollout budget. Raw traces include run manifests, resolved Hydra configurations,
trainer logs, standard rollout JSONL, and one full episode sidecar per trajectory.

## 4. Results

Values are mean plus or minus sample standard deviation across three seeds.

| Method | Final success | Paired uplift vs GRPO | Extra rollout tokens | Valid probes | High-impact / 1k tokens |
|---|---:|---:|---:|---:|---:|
| GRPO | 24.0% +/- 2.0% | - | 0.0% | 0/0 | - |
| Random-B2 | 29.2% +/- 4.6% | +5.2 +/- 6.5 pp | 51.2% | 1140/1200 | 11.93 |
| Surprisal-B2 | 31.2% +/- 3.4% | +7.3 +/- 4.3 pp | 57.3% | 1142/1200 | 10.44 |
| LinearUCB-B2 | 31.0% +/- 1.2% | +7.0 +/- 0.8 pp | 51.7% | 1138/1200 | 11.32 |

LinearUCB improves over GRPO in all three matched seeds: 9, 10, and 8 additional successful test
episodes out of 128. Random has one large gain, one small gain, and one tie, producing substantially
higher variance. Surprisal has the highest numerical mean by 0.2 points, but this difference is
negligible compared with seed variation and comes with 5.6 points more rollout overhead than
LinearUCB.

The scheduler diagnostic does not support the original hypothesis that LinearUCB discovers more
high-impact anchors per probe token. Random scores `11.93` versus `11.32` for LinearUCB. A plausible
interpretation is that the binary high-impact count ignores delta sign, training relevance, and
where the turn occurs, while LinearUCB's more consistent credits still reduce outcome variance.
That interpretation remains a hypothesis, not a measured causal explanation.

## 5. Engineering failures and fixes

The result required several failures to be isolated and documented:

- Hugging Face downloads failed in AutoDL no-card mode; network acceleration, disabled Xet, and a
  persistent cache made preparation restartable.
- Full-vocabulary entropy exhausted GPU memory with an 11.97 GiB allocation; chunked entropy,
  bounded token batches, and a shorter response cap fixed it.
- Colocated vLLM reservation killed the FSDP actor during weight synchronization; lowering vLLM
  utilization from 0.45 to 0.25 restored transient headroom.
- Verl padded 16 real trajectories to 32 because of an incompatible PPO mini-batch multiple; the
  final mini-batch size of four matches 16 real rows and avoids fabricating probe sidecars.
- A failed Random-B2 directory remains preserved and is excluded only because it lacks final-step
  sidecars. The summarizer tests this rule.

The full chronology and commands are in
`experiments/environment/2026-09-21-public-ablation-incidents.md`.

## 6. Limitations

1. Three seeds are enough to show repeatability for a portfolio, but not enough for a formal
   significance claim.
2. LinearUCB exceeds the original `1.5x` rollout-token ceiling by 1.7 percentage points.
3. The current probe budget is episode-first rather than a true group-wide top-B decision.
4. Binary Sokoban terminal reward makes many counterfactual branches reward-equivalent and gives a
   coarse scheduler target.
5. Prefix replay requires deterministic, enumerable-action environments; browser environments need
   additional session and hidden-state controls.
6. A local factual-minus-one-alternative delta is not the full causal contribution of a turn.
7. Seed 17's retained GRPO arm predates the final mini-batch padding fix. Its synthetic rows carried
   zero loss masks, but rerunning that arm is required for paper-level configuration identity.

## 7. Reproduction and portfolio use

Run CPU correctness checks with `make check`. Generate the committed aggregate and vector figure
with `make results`. Formal GPU runs use `scripts/run_sokoban_ablation.sh`; exact commands and paths
are described in `experiments/README.md`.

The strongest honest resume claim is the stable paired LinearUCB result, not scheduler dominance:

> Built ProbeGRPO, a Qwen3.5 Agent-RL system on current verl with deterministic counterfactual
> replay, budget-aware anchor scheduling, and turn-local advantage shaping; improved public
> Sokoban success from 24.0% +/- 2.0% to 31.0% +/- 1.2% across three seeds while measuring a
> 1.517x rollout-token cost.
