#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/RAGEN" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAGEN_DIR="$1"
CONFIG_NAME="probegrpo_sokoban"
CONFIG_TARGET="$RAGEN_DIR/config/${CONFIG_NAME}.yaml"

if [[ "${PROBEGRPO_SKIP_COMPAT_CHECK:-0}" != "1" ]]; then
  bash "$PROJECT_DIR/scripts/check_ragen_compat.sh" "$RAGEN_DIR"
fi

cp "$PROJECT_DIR/configs/qwen3_5_2b_sokoban.yaml" "$CONFIG_TARGET"
cd "$RAGEN_DIR"
PYTHONPATH="$PROJECT_DIR/src:$RAGEN_DIR" python3 train.py \
  --config-name "$CONFIG_NAME" \
  trainer.total_training_steps=5 \
  trainer.val_before_train=false \
  trainer.logger='[console]'
