#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [main|lambda|budget]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$1"
PHASE="${2:-main}"
SEED="${SEED:-17}"
RUN_TAG="${RUN_TAG:-$(git -C "$PROJECT_DIR" rev-parse --short HEAD)}"
ROOT_OUTPUT="${ROOT_OUTPUT:-$PROJECT_DIR/outputs/ablation-${RUN_TAG}-seed${SEED}}"
EXPECTED_FINAL_STEP="${TOTAL_TRAINING_STEPS:-50}"

case "$PHASE" in
  main)
    ARMS=(
      "grpo:0:random:0.5"
      "random_b2:2:random:0.5"
      "surprisal_b2:2:surprisal:0.5"
      "linear_ucb_b2:2:linear_ucb:0.5"
    )
    ;;
  lambda)
    ARMS=(
      "linear_ucb_b2_l025:2:linear_ucb:0.25"
      "linear_ucb_b2_l100:2:linear_ucb:1.0"
    )
    ;;
  budget)
    ARMS=(
      "linear_ucb_b1:1:linear_ucb:0.5"
      "linear_ucb_b4:4:linear_ucb:0.5"
    )
    ;;
  *)
    echo "Unknown phase '$PHASE'; expected main, lambda, or budget." >&2
    exit 2
    ;;
esac

for specification in "${ARMS[@]}"; do
  IFS=: read -r method budget scheduler lambda_coef <<<"$specification"
  output="$ROOT_OUTPUT/$method"
  if [[ -d "$output/rollouts/episodes/step-$EXPECTED_FINAL_STEP" ]]; then
    echo "==> $method already completed step $EXPECTED_FINAL_STEP; skipping"
    continue
  fi
  echo "==> $method (budget=$budget scheduler=$scheduler lambda=$lambda_coef)"
  METHOD="$method" \
  SEED="$SEED" \
  PROBE_BUDGET="$budget" \
  PROBE_SCHEDULER="$scheduler" \
  PROBE_LAMBDA="$lambda_coef" \
  OUTPUT_DIR="$output" \
    bash "$PROJECT_DIR/scripts/run_public_sokoban_train.sh" "$VERL_DIR"
done

"$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/summarize_sokoban_ablation.py" \
  "$ROOT_OUTPUT"
