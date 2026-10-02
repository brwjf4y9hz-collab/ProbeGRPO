#!/usr/bin/env bash
# 这里只准备固定版本源码；bootstrap_verl.sh 还会安装 GPU 运行环境。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${1:-${VERL_DIR:-$PROJECT_DIR/../probegrpo-runtime/verl}}"
PINNED_COMMIT="$(cat "$PROJECT_DIR/repro/verl.commit")"
PINNED_REMOTE="$(cat "$PROJECT_DIR/repro/verl.remote")"
PATCH_FILE="$PROJECT_DIR/patches/verl-probegrpo.patch"
TARGET=verl/trainer/ppo/v1/trainer_base.py
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
  case "$changed" in "$TARGET"|.gitignore|'') ;; *)
    echo "Unexpected tracked framework modification: $changed" >&2; exit 1;;
  esac
done < <(git -C "$VERL_DIR" diff --name-only HEAD)
case "$(git hash-object "$VERL_DIR/$TARGET")" in
  "$PATCHED_BLOB") echo "Pinned ProbeGRPO patch already present" ;;
  "$UPSTREAM_BLOB")
    git -C "$VERL_DIR" apply --check "$PATCH_FILE"
    git -C "$VERL_DIR" apply "$PATCH_FILE"
    ;;
  *) echo "Unrecognized trainer contents; preserving existing files" >&2; exit 1 ;;
esac
[[ "$(git hash-object "$VERL_DIR/$TARGET")" == "$PATCHED_BLOB" ]]
echo "Verified verl $PINNED_COMMIT with the exact ProbeGRPO trainer patch: $VERL_DIR"
