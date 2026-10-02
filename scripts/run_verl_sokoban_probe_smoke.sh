#!/usr/bin/env bash
# 本脚本执行 run verl sokoban probe smoke 对应的仓库流程；具体参数、环境变量和命令保持原样。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [Hydra overrides...]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/verl-sokoban-probe-smoke}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export PROBEGRPO_DEBUG_PROBE=1
PROBE_OVERRIDE='+ray_kwargs.ray_init.runtime_env.env_vars.PROBEGRPO_DEBUG_PROBE="1"'

# 如果 Hydra 再次向 Ray 传递整数型资源值，就在加载模型权重前终止。
"$1/.venv/bin/python" -c '
import sys
from omegaconf import OmegaConf
value = OmegaConf.from_dotlist([sys.argv[1][1:]]).ray_kwargs.ray_init.runtime_env.env_vars.PROBEGRPO_DEBUG_PROBE
if value != "1" or not isinstance(value, str):
    raise SystemExit("Probe debug runtime env must be a string")
' "$PROBE_OVERRIDE"

# 此验证流程测量成对后缀，但有意保持 GRPO 优势不变。
bash "$PROJECT_DIR/scripts/run_verl_sokoban_smoke.sh" "$@" \
  "$PROBE_OVERRIDE"
