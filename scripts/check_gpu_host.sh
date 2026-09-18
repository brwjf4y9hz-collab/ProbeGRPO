#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EXISTING_WORKSPACE_DIRECTORY" >&2
  exit 2
fi

WORKSPACE_DIR="$1"
MIN_GPU_MEMORY_MIB=45000
MIN_DISK_GIB=80
MIN_CUDA_VERSION="12.8"

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "Workspace directory does not exist: $WORKSPACE_DIR" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ProbeGRPO GPU training requires Linux; found $(uname -s)." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Start an NVIDIA GPU instance first." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to bootstrap uv." >&2
  exit 1
fi

echo "Detected GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

MAX_MEMORY_MIB="$({
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
} | sort -nr | head -n 1)"
if [[ -z "$MAX_MEMORY_MIB" || "$MAX_MEMORY_MIB" -lt "$MIN_GPU_MEMORY_MIB" ]]; then
  echo "Need a GPU with at least ${MIN_GPU_MEMORY_MIB} MiB; found ${MAX_MEMORY_MIB:-unknown}." >&2
  exit 1
fi

CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "$CUDA_VERSION" ]]; then
  echo "Could not read the CUDA compatibility level from nvidia-smi." >&2
  exit 1
fi
LOWEST_VERSION="$(printf '%s\n' "$MIN_CUDA_VERSION" "$CUDA_VERSION" | sort -V | head -n 1)"
if [[ "$LOWEST_VERSION" != "$MIN_CUDA_VERSION" ]]; then
  echo "Need CUDA compatibility >= ${MIN_CUDA_VERSION}; driver reports ${CUDA_VERSION}." >&2
  exit 1
fi

AVAILABLE_DISK_GIB="$(df -Pk "$WORKSPACE_DIR" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if [[ "$AVAILABLE_DISK_GIB" -lt "$MIN_DISK_GIB" ]]; then
  echo "Need at least ${MIN_DISK_GIB} GiB free under $WORKSPACE_DIR; found ${AVAILABLE_DISK_GIB}." >&2
  exit 1
fi

echo "GPU host check passed"
echo "CUDA compatibility: $CUDA_VERSION"
echo "Largest GPU memory: ${MAX_MEMORY_MIB} MiB"
echo "Workspace free disk: ${AVAILABLE_DISK_GIB} GiB"
