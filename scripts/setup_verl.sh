#!/usr/bin/env bash
# 这里只准备固定版本源码；bootstrap_verl.sh 还会安装 GPU 运行环境。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${1:-${VERL_DIR:-$PROJECT_DIR/../probegrpo-runtime/verl}}"
PINNED_COMMIT="$(cat "$PROJECT_DIR/repro/verl.commit")"
PINNED_REMOTE="$(cat "$PROJECT_DIR/repro/verl.remote")"
PATCH_FILE="$PROJECT_DIR/patches/verl-probegrpo.patch"
TARGET=verl/trainer/ppo/v1/trainer_base.py
ENGINE_TARGET=verl/workers/engine/fsdp/transformer_impl.py
UPSTREAM_BLOB=f5eebc6cdee9b076c41d10693762b57e4f5c750c
PATCHED_BLOB=8bb3a1a6676c6b53cd29c606c1d10ed8bc552f14

if [[ ! -e "$VERL_DIR" ]]; then
  mkdir -p "$VERL_DIR"
  git -C "$VERL_DIR" init -q
  git -C "$VERL_DIR" remote add origin "$PINNED_REMOTE"
  git -C "$VERL_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
  git -C "$VERL_DIR" checkout --detach FETCH_HEAD
fi
VERL_DIR="$(cd "$VERL_DIR" && pwd -P)"
if [[ "$(git -C "$VERL_DIR" rev-parse --show-toplevel)" != "$VERL_DIR" ]] || \
   [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  echo "Expected a standalone verl checkout at $PINNED_COMMIT: $VERL_DIR" >&2
  exit 1
fi
while IFS= read -r changed; do
  case "$changed" in "$TARGET"|"$ENGINE_TARGET"|.gitignore|'') ;; *)
    echo "Unexpected tracked framework modification: $changed" >&2; exit 1;;
  esac
done < <(git -C "$VERL_DIR" diff --name-only HEAD)
# Check both targets before changing either one; never overwrite an unknown engine.
case "$(git hash-object "$VERL_DIR/$ENGINE_TARGET")" in
  c5cbc35dbeeee6127c75374d70e24daaa1efd86f|1438f69c73f33ca5183c113bf01f47811274f8a4) ;;
  *) echo "Unrecognized FSDP engine contents; preserving existing files" >&2; exit 1 ;;
esac
PATCH_PYTHON="$VERL_DIR/.venv/bin/python"
if [[ ! -x "$PATCH_PYTHON" ]]; then
  PATCH_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PATCH_PYTHON" ]]; then
  echo "Python 3 is required to verify the entropy patch; no files changed" >&2
  exit 2
fi
case "$(git hash-object "$VERL_DIR/$TARGET")" in
  "$PATCHED_BLOB") echo "Pinned ProbeGRPO patch already present" ;;
  "$UPSTREAM_BLOB")
    git -C "$VERL_DIR" apply --check "$PATCH_FILE"
    git -C "$VERL_DIR" apply "$PATCH_FILE"
    ;;
  *) echo "Unrecognized trainer contents; preserving existing files" >&2; exit 1 ;;
esac
[[ "$(git hash-object "$VERL_DIR/$TARGET")" == "$PATCHED_BLOB" ]]
"$PATCH_PYTHON" "$PROJECT_DIR/scripts/install_verl_dense_entropy.py" "$VERL_DIR"
echo "Verified verl $PINNED_COMMIT with the exact ProbeGRPO trainer patch: $VERL_DIR"
