# Artifact catalog — preservation 2026-10-02

## Public versioned evidence

Release tag: `repro-2026-10-02-v1` in the
[verified release](https://github.com/brwjf4y9hz-collab/ProbeGRPO/releases/tag/repro-2026-10-02-v1).
The release is the distribution point for the following separately checksummed assets.
See `release-assets.json` for exact sizes and SHA-256, and verify `SHA256SUMS` before use.

| Asset | Contents | How to use |
|---|---|---|
| `experiment-evidence.tar.gz` | Original logs, rollout/probe sidecars and run manifests; excludes checkpoints and Ray session files | Restore under an artifact root and run `scripts/reproduce_results.py` |
| `evidence-manifest.json` | Per-file SHA-256 for all 20,399 evidence entries | `scripts/artifact_manifest.py verify-archive` |
| `verl-upstream-cf14ded.tar.gz` | Upstream source archive at the pinned commit, including original license and dependency lock; no `.git` | Offline source reference/backup; normal installer uses the fixed Git commit |
| `SHA256SUMS` | Distribution checksums | `sha256sum -c SHA256SUMS` (Linux), `shasum -a 256 -c SHA256SUMS` (macOS) |

Release source code contains the exact ProbeGRPO patch, immutable model/data references,
observed runtime versions and reproduction instructions. A source archive is not a
preinstalled environment. Retain upstream licenses when redistributing framework/model/data.

## Owner backups outside Git

- Original AutoDL project and outputs remain in their original persistent directory.
- Local `ProbeGRPO-archive-20261002/` contains evidence, provenance and data archives,
  the full-output inventory, and the complete checkpoint backup, verified against all 20,412 manifest entries.
  Its `BACKUP_STATUS.json` is the authority for completion; `.part` means incomplete.
- `provenance.tar.gz` preserves the original project Git bundle, unmodified upstream
  framework archive, full local framework diff and actual training package observations.
- `data-snapshot.tar.gz` preserves nine raw/prepared input files plus its manifest;
  public reproduction downloads the original fixed source revision and rebuilds the splits.
  The project license does not grant a new license to third-party dataset contents.
- The independently verified Qwen base snapshot is retained in the owner's earlier
  `ProbeGRPO-data-backup` directory; all 13 expected files are listed in `model-manifest.json`.

The large full-output backup is not a Git object and is not advertised as a public model
release. It contains the early GSM8K step-6 FSDP checkpoint and optimizer state. Main
Sokoban final weights do not exist because those runs disabled checkpoint saving.

## Recovery order

1. Checkout the preserved code tag; run CPU checks.
2. Reconstruct the pinned framework and environment using `REPRODUCE.md`.
3. Download/verify the fixed model and dataset, or restore the owner input backups.
4. Download the small evidence archive to inspect and recompute the visible historical results.
5. Restore full outputs only for original logs/checkpoint investigation; verify against
   `outputs-manifest.json`. An old Ray `session_latest` symlink records an obsolete absolute
   target and is not required to reproduce results.

Do not delete original server data until at least one independent complete backup has
passed manifest verification. Keep backups separate from environments that may be rebuilt.
