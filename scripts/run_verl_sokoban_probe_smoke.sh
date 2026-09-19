#!/usr/bin/env bash
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

# Fail before loading model weights if Hydra would hand Ray an integer again.
"$1/.venv/bin/python" -c '
import sys
from omegaconf import OmegaConf
value = OmegaConf.from_dotlist([sys.argv[1][1:]]).ray_kwargs.ray_init.runtime_env.env_vars.PROBEGRPO_DEBUG_PROBE
if value != "1" or not isinstance(value, str):
    raise SystemExit("Probe debug runtime env must be a string")
' "$PROBE_OVERRIDE"

# This gate measures paired suffixes but deliberately leaves GRPO advantages unchanged.
bash "$PROJECT_DIR/scripts/run_verl_sokoban_smoke.sh" "$@" \
  "$PROBE_OVERRIDE"
