#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/verl [additional Hydra overrides...]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$1"
shift

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
DATA_DIR="${DATA_DIR:-$VERL_DIR/data/probegrpo-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/verl-grpo-smoke}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"

# Set these before the compatibility check (which also loads the model config).
export HF_HOME="${HF_HOME:-$VERL_DIR/data/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Ray appends session/socket names; a project-local path exceeds AF_UNIX's limit.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pgr}"

bash "$PROJECT_DIR/scripts/check_verl_stack.sh" "$VERL_DIR"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR/logs"

cd "$VERL_DIR"
VERL_PYTHON="$VERL_DIR/.venv/bin/python"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  "$VERL_PYTHON" examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-$VERL_DIR/data/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VERL_DIR")/cache/uv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

"$VERL_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=2 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.prompt_length=256 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  trainer.use_v1=True \
  trainer.logger='["console"]' \
  trainer.project_name=probegrpo \
  trainer.experiment_name=qwen3.5-2b-gsm8k-grpo-smoke \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=5 \
  trainer.test_freq=-1 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  ray_kwargs.ray_init.runtime_env.py_executable="$VERL_PYTHON" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/logs/train-${TOTAL_TRAINING_STEPS}-steps.log"
