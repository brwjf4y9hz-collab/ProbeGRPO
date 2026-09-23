#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [Hydra overrides...]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/verl-sokoban-grpo-control}"

"$1/.venv/bin/python" "$PROJECT_DIR/scripts/install_verl_v1_probe_hook.py" "$1"

bash "$PROJECT_DIR/scripts/run_verl_sokoban_smoke.sh" "$@" \
  trainer.experiment_name=qwen3.5-2b-sokoban-grpo-control \
  +probe.enabled=true \
  +probe.budget=0 \
  +probe.scheduler=random \
  +probe.lambda_coef=0.5
