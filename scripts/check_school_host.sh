#!/usr/bin/env bash
# Read-only inventory. No torch import, model load, driver update, process kill or GPU allocation.
set -euo pipefail
WORKSPACE="${1:?Usage: bash scripts/check_school_host.sh /absolute/assigned/workspace}"
if [[ "$WORKSPACE" != /* || ! -d "$WORKSPACE" || ! -w "$WORKSPACE" ]]; then
  echo "Workspace must be an existing writable absolute directory assigned to you." >&2
  exit 2
fi
hostname
id
if [[ -d "${HOME:-/nonexistent-user-home}" ]]; then
  echo "Login home exists."
else
  echo "Warning: login home is missing; ask the administrator to repair it before SSH-key/cache setup." >&2
fi
df -h "$WORKSPACE"
free -h
git --version
python3 --version
for executable in uv conda nvcc tmux srun; do
  command -v "$executable" || true
done
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
nvidia-smi topo -m
echo "Read-only inventory complete. Idle GPUs are not an allocation; confirm assigned indices before training."
