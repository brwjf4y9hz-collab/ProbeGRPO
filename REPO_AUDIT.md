# Repository and storage audit — 2026-10-02

## Scope and Git state

Live GitHub and the original AutoDL server were inspected. At intake, `origin/main`
was `00b802e9f24e18973c4b8e45b05195865c34e5eb`; the server checkout and
`origin/codex/verl-gpu-bootstrap` were `0d3ef33424fea24661652db638d0680e4619595f`.
The server project had no tracked modifications or nonignored untracked files.
Ignored data, outputs, environments and framework changes did exist. The preservation
branch incorporates the framework provenance commit before making the changes below.
No original server run directory or existing branch history was deleted or rewritten.

## What existed where

| Material | GitHub at intake | Original server / local backup | Treatment |
|---|---|---|---|
| Project source, tests, configs, scripts, experiment notes, result CSV/SVG | Present | Present | Git; retain established directories and working import paths |
| Current framework source | Not vendored at HEAD | Separate runtime checkout | Immutable upstream SHA + small reviewed patch + setup script |
| Framework local changes | Historical diff on branch | Trainer hook plus `.gitignore`; untracked backup/data | Canonical trainer patch; full original diff retained in historical archive |
| Actual training package versions | Missing (old freeze was Conda base) | Training `.venv` available | Capture `repro/runtime-observed.txt` and JSON; use upstream lock + explicit override to install |
| Raw trajectories and logs | Ignored | 20,399 evidence files recovered | Separate evidence archive and file hashes; rebuild summaries from these records |
| All outputs including early checkpoint | Ignored | 9,982,019,944 logical bytes; 20,411 regular files + one symlink | Full separate backup; see artifact catalog for transfer/verification status |
| Main experiment final weights | Absent | **Never saved** (`save_freq=-1`) | Cannot recover; rerun training. New launcher saves the final checkpoint by default |
| Early GSM8K GRPO step-6 checkpoint | Absent | Present, ~9.4 GB with optimizer | Full output archive; this is not a main-result model or a portable LoRA adapter |
| Qwen base model | Absent | Complete 13-file fixed snapshot | Upstream immutable revision + SHA-256 manifest; independent local backup |
| RAGEN raw and prepared data | Absent | Nine files captured | Pinned source downloads + checksums + preprocessing script; separate backup |
| Environments, caches, Ray sessions, bytecode | Ignored | Present | Rebuild; do not commit |
| Prior recovery bundles and exports | Not runtime source | Present | Keep in backup storage; do not mix into source tree |

The inspection covers this server, live remote branches and identified local backups.
It cannot prove that unrelated laptops or other servers contain no additional edits.

## Directory policy

- `src/`, `tests/`, `configs/`, `scripts/`: maintained implementation and executable workflows.
- `docs/`, `experiments/`: explanations, protocols and small visible results.
- `patches/`: only ProbeGRPO's framework modification.
- `repro/`: immutable pins, manifests, observed environment and validation records.
- `repro/archive-20260929/`: old provenance retained with its limitations labeled.
- `outputs/`, `models/`, `data/`, caches, checkpoints, archives: ignored storage outside Git.

No submodule is required. The installer creates a separate pinned upstream checkout,
checks its identity and exact trainer content, and applies the patch once. A full
vendored framework would obscure the modification and duplicate upstream history.
Older Git history may still contain the previously vendored source; this cleanup
changes the current tree without destructive history rewriting. Weights and archives
are excluded from new commits; model artifacts remain external.

## Reproduction judgment

CPU flow, existing result reconstruction, model integrity, framework patch application,
and deterministic prepared-data reconstruction have distinct verification records.
**A fresh GPU environment and a fresh training run have not been demonstrated in this
cleanup.** The server was in CPU maintenance mode without an allocated GPU. Historical
training evidence is retained and matched; it is not a fresh training reproduction.
See [REPRODUCE.md](REPRODUCE.md) and [validation](repro/validation/README.md).
