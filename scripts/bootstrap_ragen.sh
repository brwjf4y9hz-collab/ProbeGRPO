#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 WORKSPACE_DIRECTORY" >&2
  exit 2
fi

WORKSPACE="$1"
RAGEN_DIR="$WORKSPACE/RAGEN"
RAGEN_COMMIT="d97bb3284e99568adfd44ee15c736d7685c07512"

if [[ -e "$RAGEN_DIR" ]]; then
  echo "Refusing to overwrite existing path: $RAGEN_DIR" >&2
  exit 1
fi

mkdir -p "$WORKSPACE"
git clone https://github.com/mll-lab-nu/RAGEN.git "$RAGEN_DIR"
git -C "$RAGEN_DIR" checkout "$RAGEN_COMMIT"
git -C "$RAGEN_DIR" submodule update --init --recursive

echo "RAGEN checked out at $RAGEN_COMMIT"
echo "Run: bash scripts/check_ragen_compat.sh '$RAGEN_DIR'"

