#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_RUNTIME_DIRECTORY" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$1"
VERL_DIR="$RUNTIME_DIR/verl"
VERL_COMMIT="${VERL_COMMIT:-cf14ded3a448107e70a206fd201817cc1cbae348}"

if [[ -e "$RUNTIME_DIR" ]]; then
  echo "Refusing to overwrite existing runtime directory: $RUNTIME_DIR" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
mkdir -p "$VERL_DIR"
git -C "$VERL_DIR" init
git -C "$VERL_DIR" remote add origin https://github.com/verl-project/verl.git
git -C "$VERL_DIR" fetch --depth 1 origin "$VERL_COMMIT"
git -C "$VERL_DIR" checkout --detach FETCH_HEAD

# Keep the large wheel cache on the persistent data disk rather than AutoDL's 30 GB system disk.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_DIR/cache/uv}"
mkdir -p "$UV_CACHE_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  python3 -m pip install --user uv
  USER_BASE="$(python3 -m site --user-base)"
  UV_BIN="$USER_BASE/bin/uv"
fi

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv installation failed: $UV_BIN is not executable" >&2
  exit 1
fi

cd "$VERL_DIR"
"$UV_BIN" sync --frozen --all-packages --extra vllm --extra fsdp
"$UV_BIN" pip install --python "$VERL_DIR/.venv/bin/python" --no-deps -e "$PROJECT_DIR"

cat > "$RUNTIME_DIR/runtime_manifest.txt" <<EOF
probegrpo_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
verl_commit=$(git -C "$VERL_DIR" rev-parse HEAD)
uv=$($UV_BIN --version)
uv_cache=$UV_CACHE_DIR
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "verl runtime created at $VERL_DIR"
echo "Manifest: $RUNTIME_DIR/runtime_manifest.txt"
echo "Next: bash $PROJECT_DIR/scripts/check_verl_stack.sh $VERL_DIR"
