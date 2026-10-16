#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [Hydra overrides...]" >&2
  exit 2
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$1"
shift
if ! "$VERL_DIR/.venv/bin/python" -c 'import swanlab' >/dev/null 2>&1; then
  echo "Install SwanLab in $VERL_DIR/.venv before launching this run." >&2
  exit 2
fi
export SWANLAB_ENABLED=1
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$PROJECT_DIR/outputs/swanlog}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/public-sokoban-${METHOD:-grpo}-seed${SEED:-17}}"

# Explicit override also supports preserved remote launchers predating SWANLAB_ENABLED.
bash "$PROJECT_DIR/scripts/run_public_sokoban_train.sh" "$VERL_DIR" \
  trainer.logger='["console","swanlab"]' "$@"

if [[ ! -f "$OUTPUT_DIR/curves/metrics_by_step.csv" ]]; then
  "$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/plot_training_curves.py" "$OUTPUT_DIR"
fi
