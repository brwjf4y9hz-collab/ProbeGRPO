#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [Hydra overrides...]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$1"
shift

METHOD="${METHOD:-grpo}"
SEED="${SEED:-17}"
PROBE_BUDGET="${PROBE_BUDGET:-0}"
PROBE_SCHEDULER="${PROBE_SCHEDULER:-random}"
PROBE_LAMBDA="${PROBE_LAMBDA:-0.5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
TEST_LIMIT="${TEST_LIMIT:-128}"
MAX_TURNS="${MAX_TURNS:-12}"
DATA_DIR="${DATA_DIR:-$VERL_DIR/data/probegrpo-ragen-public-sokoban-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/public-sokoban-${METHOD}-seed${SEED}}"
MODEL_REVISION="${MODEL_REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
export HF_HOME="${HF_HOME:-$VERL_DIR/data/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
MODEL_PATH="${MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen3.5-2B/snapshots/$MODEL_REVISION}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Pinned model snapshot is missing: $MODEL_PATH" >&2
  echo "Download Qwen/Qwen3.5-2B revision $MODEL_REVISION before starting paid training." >&2
  exit 2
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse output directory: $OUTPUT_DIR" >&2
  echo "Set OUTPUT_DIR to a new path so sidecars and logs cannot mix." >&2
  exit 2
fi

export DATA_DIR MODEL_PATH OUTPUT_DIR TOTAL_TRAINING_STEPS
export PYTHONHASHSEED="$SEED"
mkdir -p "$DATA_DIR"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  "$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/prepare_public_sokoban_data.py" \
    --output-dir "$DATA_DIR" \
    --train-limit "$TRAIN_LIMIT" \
    --test-limit "$TEST_LIMIT" \
    --max-turns "$MAX_TURNS"
fi

"$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/install_verl_v1_probe_hook.py" "$VERL_DIR"
"$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/write_run_manifest.py" \
  --project-dir "$PROJECT_DIR" \
  --verl-dir "$VERL_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --method "$METHOD" \
  --seed "$SEED" \
  --budget "$PROBE_BUDGET" \
  --scheduler "$PROBE_SCHEDULER" \
  --lambda-coef "$PROBE_LAMBDA" \
  --model "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" \
  --steps "$TOTAL_TRAINING_STEPS"

bash "$PROJECT_DIR/scripts/run_verl_sokoban_smoke.sh" "$VERL_DIR" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.val_before_train=true \
  trainer.test_freq=10 \
  trainer.experiment_name="public-sokoban-${METHOD}-seed${SEED}" \
  data.train_batch_size=4 \
  data.val_batch_size="$TEST_LIMIT" \
  data.dataloader_num_workers=0 \
  data.max_response_length=1024 \
  data.shuffle=true \
  data.seed="$SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.entropy_from_logits_chunk_size=256 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.seed="$SEED" \
  +probe.enabled=true \
  +probe.budget="$PROBE_BUDGET" \
  +probe.scheduler="$PROBE_SCHEDULER" \
  +probe.lambda_coef="$PROBE_LAMBDA" \
  +probe.warmup_updates=20 \
  +probe.warmup_probes=200 \
  +probe.exploration_rate=0.1 \
  +probe.ucb_alpha=1.0 \
  +probe.ucb_l2=1.0 \
  "$@"
