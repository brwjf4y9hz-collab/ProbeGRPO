#!/usr/bin/env python3
"""Verify an isolated local clone; never claim a working-tree snapshot is a public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_FILES = {".gitignore", ".gitattributes", "LICENSE", "Makefile", "README.md", "REPORT.md", "MODEL_CARD.md",
              "AGENTS.md", "pyproject.toml", "environment.yml", "REPRODUCE.md", "REPRODUCIBILITY.md",
              "REPO_AUDIT.md", "PATCHES.md"}
DIRECTORIES = {".github", "src", "tests", "scripts", "configs", "docs", "experiments", "patches", "repro", "data"}
EXTENSIONS = {".py", ".sh", ".toml", ".yml", ".yaml", ".md", ".json", ".csv", ".svg", ".png", ".pdf",
              ".patch", ".commit", ".remote", ".txt"}


def source_files(root):
    output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"])
    selected = []
    for name in sorted(set(os.fsdecode(output).split("\0")) - {""}):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source path escapes repository")
        if not (name in ROOT_FILES or (relative.parts[0] in DIRECTORIES and relative.suffix in EXTENSIONS)):
            continue
        if "__pycache__" in relative.parts or relative.name.startswith(".") and name not in ROOT_FILES:
            continue
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"refusing symlink or escaped source: {name}")
        if path.is_file():
            selected.append(relative)
    return selected


def clean_env():
    # Do not pass cloud credentials, editable installs, or user PYTHONPATH to the clone.
    return {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
            "PYTHONNOUSERSITE": "1", "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "",
            "SWANLAB_ENABLED": "0", "SWANLAB_MODE": "disabled",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TEMPLATE_DIR": ""}


def verify(root, output, mode):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("use a new evidence directory")
    output.mkdir(parents=True)
    env = clean_env()
    source_revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    records, file_hashes = [], {}

    def run(command, cwd):
        number = len(records) + 1
        completed = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log = output / f"command-{number:02d}.log"
        # Git may emit harmless macOS temp-directory warnings on stderr. Do not
        # interpret stderr as porcelain status or contaminate JSON plan parsing.
        log.write_text(completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""))
        records.append({"command": command, "cwd": str(cwd), "exit_code": completed.returncode,
                        "log": log.name, "sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        if completed.returncode:
            raise RuntimeError(f"command failed; see {log}")
        return completed.stdout

    payload = {"mode": mode, "source_revision": source_revision, "public_clone_verified": False,
               "gpu_training": "not_run", "status": "running", "commands": records,
               "source_files_sha256": file_hashes}
    try:
        source = root
        if mode == "working-tree":
            source = output / "snapshot"
            source.mkdir()
            for relative in source_files(root):
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / relative, target)
                file_hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
            run(["git", "init", "-b", "cpu-snapshot", str(source)], output)
            run(["git", "add", "."], source)
            run(["git", "-c", "user.name=ProbeGRPO local verifier", "-c", "user.email=verifier@localhost",
                 "-c", "commit.gpgsign=false", "commit", "-m", "Temporary working-tree CPU snapshot; not published"], source)
        clone = output / "checkout"
        run(["git", "clone", "--no-local", "--no-hardlinks", str(source), str(clone)], output)
        before = run(["git", "status", "--porcelain"], clone)
        if before.strip():
            raise ValueError("clone is not clean before verification")
        run([sys.executable, "-m", "probegrpo.smoke"], clone)
        run([sys.executable, "-m", "probegrpo.agent_smoke"], clone)
        run(["make", "check"], clone)
        # Plan mode must not require a real runtime, torch, credentials or GPUs.
        for profile in ("single_48gb", "single_24gb", "dual_24gb"):
            plan = json.loads(run([sys.executable, "scripts/run_final_controls.py", "--verl-dir",
                                  "/nonexistent/verl-runtime", "--hardware-profile", profile], clone))
            if plan["total_optimizer_updates"] != 2400 or len(plan["runs"]) != 12:
                raise ValueError("unexpected default scientific matrix")
        after = run(["git", "status", "--porcelain"], clone)
        if after.strip():
            raise ValueError("verification changed tracked/unignored source files")
        payload["status"] = "success"
        payload["source_fingerprint"] = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest() if file_hashes else None
        payload["verified_clone_commit"] = run(["git", "rev-parse", "HEAD"], clone).strip()
    except Exception as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("committed", "working-tree"), default="committed")
    args = parser.parse_args()
    result = verify(args.repo, args.output_dir, args.mode)
    print(json.dumps({key: value for key, value in result.items() if key not in {"commands", "source_files_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
