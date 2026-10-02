# Reproduce ProbeGRPO

## What each command establishes

| Target | Inputs / hardware | Evidence and limitation |
|---|---|---|
| CPU tests and smoke | Python >=3.9; no model/GPU | Core replay, credit and data-flow checks; tensor tests need torch |
| Committed result rendering | Standard-library Python | CSV → summary/JSON/SVG; does not independently verify raw data |
| Raw result reconstruction | Preserved experiment evidence | 12 runs × 128 final evaluations, same held-out boards, 50 updates, raw costs |
| Framework reconstruction | Git/network | Exact upstream commit plus exact trainer patch; no GPU execution |
| Data reconstruction | Training environment + fixed raw files | Rebuilds 512/128 parquet splits; hashes matched recorded files |
| GPU smoke / training | Linux NVIDIA CUDA, model, data, large persistent disk | Instructions supplied; not rerun from a fresh GPU environment during cleanup |

## 1. Clean checkout and smallest smoke

```bash
git clone https://github.com/brwjf4y9hz-collab/ProbeGRPO.git
cd ProbeGRPO
# For the preserved snapshot, checkout the release tag listed in repro/ARTIFACTS.md.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
make check
make results
```

`make check` runs unit tests, CPU smoke, agent-flow smoke and shell syntax checks.
Without torch, two tensor-specific checks skip; that does not validate the GPU path.
`make results` regenerates the committed summary and figure using full-precision CSV.

## 2. Restore and verify original experiment evidence

Download the small evidence archive and `evidence-manifest.json` from the release linked
in [the artifact catalog](repro/ARTIFACTS.md), then run from the repository root:

```bash
python3 scripts/artifact_manifest.py verify-archive /path/to/experiment-evidence.tar.gz \
  --prefix outputs --manifest /path/to/evidence-manifest.json
mkdir -p artifacts/original
tar -xzf /path/to/experiment-evidence.tar.gz -C artifacts/original
python3 scripts/reproduce_results.py --artifact-root artifacts/original \
  --output-dir artifacts/recomputed
```

Inspect `artifacts/recomputed/verification.json` and `summary.md`. The command checks
all 1,536 final binary rewards, identical unique 128-board sets, source/run metadata,
step 50 and full-precision costs. It compares with the committed values, fails on
mismatch, and leaves the original evidence untouched. A result match validates these
archived observations; it does not prove a future stochastic training run gives identical scores.

## 3. Framework and actual environment

Canonical upstream pin: `repro/verl.commit` (`cf14ded3a448107e70a206fd201817cc1cbae348`).
Code modification: `patches/verl-probegrpo.patch`, 16 added lines. `.gitignore` changes
and trainer backup files are not required to reproduce the trainer behavior.

Source-only reconstruction (also usable on a CPU machine):

```bash
bash scripts/setup_verl.sh /absolute/path/to/new-verl
# Repeating this is safe: it verifies that the exact patch is already present.
bash scripts/setup_verl.sh /absolute/path/to/new-verl
```

Unexpected trainer edits or a different upstream commit cause a failure, preserving
existing files. The current setup script replaces an incomplete historical script.

On a Linux NVIDIA host, use a **new**, persistent runtime directory:

```bash
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
export VERL_DIR=/path/to/persistent-workspace/probegrpo-runtime/verl
```

The recipe uses Python 3.12.3, uv 0.12.15, upstream `uv.lock` with `vllm` and `fsdp`
extras, then the observed NumPy 2.3.5 override. It installs the project editable without
resolving dependencies and runs `uv pip check`. Actual historical training package
versions are recorded in `repro/runtime-observed.txt` / `runtime-environment.json`:
torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0, ray 2.55.1, datasets 5.0.0.
This capture is an observation, not a portable requirements lock. The older
`repro/archive-20260929/autodl-pip-freeze.txt` came from Conda base and must not be used.
A fresh GPU installation remains an open verification item.

## 4. Immutable model and dataset

```bash
export MODEL_PATH=/path/to/persistent-workspace/models/Qwen3.5-2B-15852e8
"$VERL_DIR/.venv/bin/python" scripts/download_model.py --output-dir "$MODEL_PATH"
# Offline integrity check of an existing snapshot:
python3 scripts/download_model.py --verify-only --output-dir "$MODEL_PATH"
export DATA_DIR="$VERL_DIR/data/probegrpo-ragen-public-sokoban-v1"
"$VERL_DIR/.venv/bin/python" scripts/prepare_public_sokoban_data.py --output-dir "$DATA_DIR"
```

Model: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
The downloader uses only this revision and checks all 13 recorded file hashes.
Weights alone are 4,548,221,488 bytes. Upstream model license/attribution applies.

Dataset: `ZihanWang314/ragen-datasets`, revision
`e2060cf7c51da891a68e0e7437309e5ec052e45b`. The preparation script verifies source
SHA-256, deduplicates boards, excludes train/test overlap, and computes exact shortest
solutions. Defaults are train 512, test 128, maximum 12 turns. An offline raw backup
can be used with `--source-dir /path/to/raw-ragen` containing `train.parquet` and
`test.parquet`. Original source metadata/license terms remain applicable; the project
license does not relicense third-party datasets.

Recorded/rebuilt output hashes (with the captured training package versions):

- train: `75ac2a084eba22d34d18ed214791e1775c71cb9cd739eca875d46402e224d736`
- test: `31d8b0b4ec5d8951af305289eb430686ed9e95a409fa0d43e0e6abd9d87b2728`

Different parquet library versions can change serialized bytes; use the observed stack
when comparing file hashes. Never silently substitute a new dataset revision.

## 5. GPU smoke, then the public comparison

Historical runs recorded one NVIDIA GeForce RTX 4090 48 GB. The cleanup server had no
GPU allocated. A 48 GB compatible GPU and working CUDA driver are required for this
recipe; compatibility on another GPU/driver has not been demonstrated here.

```bash
bash scripts/check_verl_stack.sh "$VERL_DIR"
OUTPUT_DIR="$PWD/outputs/grpo-gate-new" bash scripts/run_verl_grpo_smoke.sh "$VERL_DIR"
OUTPUT_DIR="$PWD/outputs/sokoban-gate-new" bash scripts/run_verl_sokoban_smoke.sh "$VERL_DIR"
# Main protocol: run only after inspecting the smoke results and available disk.
SEED=17 RUN_TAG=preserved-v1 bash scripts/run_sokoban_ablation.sh "$VERL_DIR" main
# Repeat with SEED=42 and SEED=101 for the full 12-run comparison.
```

The public launcher uses 50 updates, four methods, budget B=2 for probe arms and fixed
train/test inputs. Read `docs/SOKOBAN_AGENTLOOP.md` and the result record for the
historical source differences between seeds. The current source is a maintained
reproduction recipe, not a claim that all historical runs used the same commit.
Run manifests now capture source status, framework/launcher hashes, relevant package
versions, checkpoint frequency and user Hydra overrides. Logs contain resolved config.

**Checkpoint behavior:** historical main runs set `trainer.save_freq=-1`, so there are
no final main-result weights to download. The new launcher defaults `SAVE_FREQ` to the
final step. This preserves final weights without changing the loss or sampling setup,
but increases disk use. The existing full FSDP smoke checkpoint is about 9.4 GB; allow
at least 150 GB persistent storage for twelve similarly sized final checkpoints plus
runtime/model/logs, or archive each run before starting the next. The original 100 GB
server does not have room for all of these. `SAVE_FREQ=-1` recreates the historical
no-checkpoint setting, but again discards final weights.

The archived early GSM8K step-6 checkpoint is not the final Sokoban model and is not a
standalone LoRA adapter. Keep its model, optimizer, metadata and pinned framework
layout together when attempting resume; clean-environment resume is not yet verified.

## Known missing items / next verification

1. Fresh Linux GPU environment installation, smoke, and resume verification.
2. Fresh main-protocol training and evaluation from the pinned inputs.
3. Main experiment final model weights (never saved; require retraining).
4. Portable trained adapter export and an inference-only demonstration.

These are explicit gaps. CPU success and matching archived scores do not close them.
