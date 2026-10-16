#!/usr/bin/env bash
# 本脚本执行 bootstrap verl 对应的仓库流程；具体参数、环境变量和命令保持原样。
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--resume" ) ]]; then
  echo "Usage: $0 RUNTIME_DIRECTORY [--resume]" >&2
  exit 2
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$1"
RESUME="${2:-}"
if [[ -L "$RUNTIME_DIR" ]]; then
  echo "Refusing a symlink runtime directory: $RUNTIME_DIR" >&2
  exit 1
fi
if [[ -e "$RUNTIME_DIR" ]]; then
  if [[ "$RESUME" != "--resume" ]]; then
    echo "Refusing to overwrite existing runtime directory: $RUNTIME_DIR" >&2
    echo "For an inspected partial installation only, explicitly use --resume." >&2
    exit 1
  fi
  if [[ ! -d "$RUNTIME_DIR/verl/.git" || -e "$RUNTIME_DIR/runtime_manifest.txt" ]]; then
    echo "Resume requires a partial standalone verl checkout without a completion manifest." >&2
    exit 1
  fi
elif [[ "$RESUME" == "--resume" ]]; then
  echo "Cannot resume a missing runtime directory: $RUNTIME_DIR" >&2
  exit 1
fi
mkdir -p "$RUNTIME_DIR"
RUNTIME_DIR="$(cd "$RUNTIME_DIR" && pwd)"
VERL_DIR="$RUNTIME_DIR/verl"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_DIR/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUNTIME_DIR/python}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-$RUNTIME_DIR/bin}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RUNTIME_DIR/cache/pip}"
export HF_HOME="${HF_HOME:-$RUNTIME_DIR/cache/huggingface}"
mkdir -p "$UV_CACHE_DIR"
# Rechecks the fixed commit, exact patch, and unexpected changes before any dependency sync.
bash "$PROJECT_DIR/scripts/setup_verl.sh" "$VERL_DIR"
UV_VERSION=0.12.15
if [[ -x "$RUNTIME_DIR/bootstrap/bin/uv" ]]; then
  UV_BIN="$RUNTIME_DIR/bootstrap/bin/uv"
  if [[ "$("$UV_BIN" --version)" != "uv $UV_VERSION" ]]; then
    echo "Unexpected local uv version; preserving existing installation: $UV_BIN" >&2
    exit 1
  fi
elif command -v uv >/dev/null 2>&1 && [[ "$(uv --version)" == "uv $UV_VERSION" ]]; then
  UV_BIN="$(command -v uv)"
else
  if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
    echo "System Python lacks ensurepip. No system package or privilege change is required." >&2
    echo "Install official uv $UV_VERSION into $RUNTIME_DIR/bootstrap/bin, then use --resume." >&2
    echo "See docs/SCHOOL_GPU_RUNBOOK_ZH.md (ensurepip recovery). Existing files are preserved." >&2
    exit 2
  fi
  python3 -m venv "$RUNTIME_DIR/bootstrap"
  "$RUNTIME_DIR/bootstrap/bin/python" -m pip install "uv==$UV_VERSION"
  UV_BIN="$RUNTIME_DIR/bootstrap/bin/uv"
fi
cd "$VERL_DIR"
"$UV_BIN" sync --python 3.12.3 --frozen --all-packages --extra vllm --extra fsdp
# 保留上游依赖锁文件，以及已记录运行所用的兼容性覆盖项。
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
