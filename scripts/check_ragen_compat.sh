#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/RAGEN" >&2
  exit 2
fi

RAGEN_DIR="$1"
if [[ ! -f "$RAGEN_DIR/setup.py" || ! -f "$RAGEN_DIR/train.py" ]]; then
  echo "Not a RAGEN checkout: $RAGEN_DIR" >&2
  exit 2
fi

python3 - "$RAGEN_DIR" <<'PY'
import importlib
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
setup_text = (root / "setup.py").read_text(encoding="utf-8")
commit = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
print(f"RAGEN commit: {commit}")
print(f"Python: {sys.version.split()[0]}")

issues = []
if sys.version_info < (3, 12):
    issues.append("RAGEN's current setup recommends Python 3.12")

pin = re.search(r'vllm==([0-9.]+)', setup_text)
if pin:
    print(f"RAGEN setup.py vLLM pin: {pin.group(1)}")
    issues.append(
        "the inspected RAGEN revision pins vLLM 0.8.2, which predates Qwen3.5 support; "
        "do not launch Qwen3.5 with this unmodified dependency set"
    )

for package in ("torch", "transformers", "vllm", "verl"):
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: {version}")
    except Exception as error:
        print(f"{package}: unavailable ({type(error).__name__})")

if issues:
    print("\nCompatibility blockers:")
    for issue in issues:
        print(f"- {issue}")
    print(
        "\nUse a Qwen3.5-capable current verl/vLLM stack and install RAGEN with --no-deps, "
        "then run a five-update Sokoban smoke test before WebShop. See docs/GPU_RUNBOOK.md."
    )
    raise SystemExit(1)

print("No known static compatibility blockers found.")
PY

