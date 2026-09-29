# Reproduction and data preservation

## What is versioned

ProbeGRPO source, configs, experiment scripts, tests, result summaries, and the local `verl` modification are kept in Git. `verl` is pinned to the exact upstream commit in `repro/verl.commit`; the corresponding source is obtained from the upstream commit archive and the local modification is applied from `patches/verl-probegrpo.patch`. The dataset provenance record is `data/manifests/probegrpo-ragen-public-sokoban-v1.json`.

The old `probegrpo-runtime/` tree remains on the AutoDL disk for recovery, but is no longer tracked in this repository. The external `verl` checkout also remains in place. These folders were not removed.

## Environment

`environment.yml` describes the project Python environment. `repro/autodl-pip-freeze.txt` and `repro/python-version.txt` record the packages and interpreter observed in the AutoDL session; use them as a reference when rebuilding that machine. Install the GPU/CUDA-compatible PyTorch and `verl` requirements for your platform before running training. Model weights are intentionally fetched separately and are not stored in Git.

To prepare a pinned `verl` tree in a fresh environment (requires `curl`, `tar`, and network access):

```bash
conda env create -f environment.yml
conda activate probegrpo
pip install -e '.[dev]'
./scripts/setup_verl.sh
```

To use another destination, set `VERL_DIR=/path/to/verl` before running the setup script. It refuses to overwrite a pre-existing non-git directory and checks the pinned commit before applying the patch. The existing AutoDL checkout already has the patch; the script recognizes this state.

## Verification and runs

Run the CPU-level unit/smoke checks with:

```bash
bash scripts/smoke_test.sh
```

The short GPU training entry point is `scripts/run_verl_sokoban_probe_train_smoke.sh`; inspect its config and required model/cache paths before running. It needs the intended GPU, model weights, data access, and a configured `verl` runtime. Main experiment commands and hyperparameters are recorded under `scripts/`, `configs/`, and `experiments/`.

## Data and large outputs

The `outputs/` directory is preserved locally and excluded from Git because it contains about 19 GB of runs/checkpoints/logs. A file/size inventory from the original AutoDL state is saved in `recovery_20260928_01/outputs-manifest.tsv`; recovery material is intentionally ignored by Git. Keep that recovery directory and the AutoDL volume until the outputs have been copied to durable storage. For any source data stored outside the repository, follow its manifest's source and preparation instructions, download it locally, and verify the published checksums where available. Do not place Hugging Face caches, model weights, checkpoints, W&B runs, or temporary files in Git.

## Recovery snapshot

`recovery_20260928_01/ProbeGRPO.bundle` is a verified bundle of the ProbeGRPO repository history as found before cleanup. The directory also contains copies of the live `trainer_base.py`, its original backup, the full local patch, data manifest, and a size inventory of `outputs/`. The bundle and recovery directory are local safety copies, not source artifacts to publish.
