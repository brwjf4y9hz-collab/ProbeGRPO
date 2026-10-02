#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_RUNTIME_DIRECTORY" >&2
  exit 2
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$1"
if [[ -e "$RUNTIME_DIR" ]]; then
  echo "Refusing to overwrite existing runtime directory: $RUNTIME_DIR" >&2
  exit 1
fi
mkdir -p "$RUNTIME_DIR"
RUNTIME_DIR="$(cd "$RUNTIME_DIR" && pwd)"
VERL_DIR="$RUNTIME_DIR/verl"
bash "$PROJECT_DIR/scripts/setup_verl.sh" "$VERL_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_DIR/cache/uv}"
mkdir -p "$UV_CACHE_DIR"
UV_VERSION=0.12.15
if command -v uv >/dev/null 2>&1 && [[ "$(uv --version)" == "uv $UV_VERSION" ]]; then
  UV_BIN="$(command -v uv)"
else
  python3 -m venv "$RUNTIME_DIR/bootstrap"
  "$RUNTIME_DIR/bootstrap/bin/python" -m pip install "uv==$UV_VERSION"
  UV_BIN="$RUNTIME_DIR/bootstrap/bin/uv"
fi
cd "$VERL_DIR"
"$UV_BIN" sync --python 3.12.3 --frozen --all-packages --extra vllm --extra fsdp
# Preserve the upstream lock and the compatibility override used by the recorded run.
"$UV_BIN" pip install --python "$VERL_DIR/.venv/bin/python" --no-deps 'numpy==2.3.5'
"$UV_BIN" pip install --python "$VERL_DIR/.venv/bin/python" --no-deps -e "$PROJECT_DIR"
"$UV_BIN" pip check --python "$VERL_DIR/.venv/bin/python"
"$UV_BIN" pip freeze --python "$VERL_DIR/.venv/bin/python" > "$RUNTIME_DIR/installed-packages.txt"
cat > "$RUNTIME_DIR/runtime_manifest.txt" <<EOF
probegrpo_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
verl_commit=$(git -C "$VERL_DIR" rev-parse HEAD)
verl_patch_sha256=$(sha256sum "$PROJECT_DIR/patches/verl-probegrpo.patch" | cut -d ' ' -f 1)
uv=$($UV_BIN --version)
python=$($VERL_DIR/.venv/bin/python --version)
runtime_override=numpy==2.3.5
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "Runtime prepared: $VERL_DIR"
echo "Download the pinned model before running scripts/check_verl_stack.sh $VERL_DIR"
