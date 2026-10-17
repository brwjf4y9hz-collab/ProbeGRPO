"""Install/check the exact dense-entropy fix on the pinned, standalone verl checkout."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

VERL_COMMIT = "cf14ded3a448107e70a206fd201817cc1cbae348"
TARGET = Path("verl/workers/engine/fsdp/transformer_impl.py")
UPSTREAM_SHA256 = "79d8779fb2a0b8a1ac7b3ec85fba2521ac8cae079925a497c4366fd77f8abfbe"
PATCHED_SHA256 = "be9daa3ad741ad2539752344de01db3859bd580de6697b4719166a9c1b4dab97"
PATCH = Path(__file__).resolve().parents[1] / "patches/verl-dense-entropy.patch"
PATCH_SHA256 = "20925f05fed751502c9b1167dd0594823796da16ae271fc947568fcd475b0a73"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(verl_dir: Path, *, check_only: bool = False) -> str:
    verl_dir = verl_dir.resolve()
    root = subprocess.check_output(["git", "-C", str(verl_dir), "rev-parse", "--show-toplevel"],
                                   text=True).strip()
    revision = subprocess.check_output(["git", "-C", str(verl_dir), "rev-parse", "HEAD"],
                                       text=True).strip()
    if Path(root).resolve() != verl_dir or revision != VERL_COMMIT:
        raise RuntimeError("expected the pinned standalone verl checkout; no files changed")
    target = verl_dir / TARGET
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("missing/symlinked FSDP engine; no files changed")
    actual = sha256(target)
    if actual not in (UPSTREAM_SHA256, PATCHED_SHA256):
        raise RuntimeError("unrecognized FSDP engine contents; preserving existing edits")
    if sha256(PATCH) != PATCH_SHA256:
        raise RuntimeError("dense entropy patch checksum mismatch; no files changed")
    if actual == PATCHED_SHA256:
        return "Verified exact dense entropy patch (already installed)"
    if check_only:
        raise RuntimeError("dense entropy patch not installed; run scripts/setup_verl.sh first")
    command = ["git", "-C", str(verl_dir), "apply"]
    subprocess.run([*command, "--check", str(PATCH)], check=True)
    subprocess.run([*command, str(PATCH)], check=True)
    if sha256(target) != PATCHED_SHA256:
        raise RuntimeError("unexpected engine hash after patch; do not start training")
    return "Installed and verified exact dense entropy patch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(install(args.verl_dir, check_only=args.check_only))


if __name__ == "__main__":
    main()
