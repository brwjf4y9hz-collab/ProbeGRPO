# Validation record — 2026-10-02

| Check | Result | Evidence |
|---|---|---|
| Fresh Python 3.9 virtual environment editable install | Passed after upgrading the old bundled pip/build tools | README now explicitly upgrades pip |
| `make check` | 78 tests run: 76 passed, 2 torch-only tests skipped; both CPU smokes and shell syntax passed | `cpu-check.log` |
| Source-only framework reconstruction | Fresh fetch of exact commit and patch application passed; second run verified already present | `framework-setup.txt` |
| Unexpected trainer modification | Rejected and preserved | `framework-refusal.json` |
| Archived raw experiment records | 12 runs, 1,536 final evaluations, 128 identical unique boards, final update 50; exact success counts and costs within 1e-12 | `result-verification.json` |
| Evidence archive | All 20,399 entries matched the server file manifest | `evidence-verification.json` |
| Full output backup including early checkpoint | All 20,412 entries matched; 5,015,429,285 compressed bytes | Owner backup: `full-backup-verification.json` |
| Model backup | All 13 pinned snapshot files matched SHA-256 | `model-verification.json` |
| Data backup | All nine files matched | `data-backup-verification.json` |
| Prepared public data rebuilt in original training environment | Both train/test parquet hashes matched recorded files | `data-rebuild.json` |
| Fresh GPU installation, training and checkpoint resume | Not run; no GPU allocated | Explicit open items in `../../REPRODUCE.md` |

The first data archive check rejected its additional `data-manifest.json` metadata
member; the extracted `data/` tree was then verified independently against that manifest.
This was an archive-layout issue, not a data checksum mismatch.

Original rounded CSV is retained in `experiments/results/public_sokoban_main_v1/legacy/`.
Using raw precision changes displayed LinearUCB cost from 51.7% to 51.6%; reward counts
and success rates do not change. The original experiment logs remain unchanged.

CPU checks do not establish GPU training reproducibility. Historical main runs disabled
checkpoint saving; no main-run final model is available. The early GRPO step-6 checkpoint
must not be presented as a trained Sokoban result.

GitHub Actions on commit `1296593` passed all four push/PR checks on Python 3.9 and 3.12, including fresh framework setup.
