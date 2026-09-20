#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/verl [Hydra overrides...]" >&2
  exit 2
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="$1"
shift
export DATA_DIR="${DATA_DIR:-$VERL_DIR/data/probegrpo-sokoban-fixtures}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/verl-sokoban-smoke}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
# Ray may otherwise assign zero OMP threads to a fractional-CPU vLLM actor.
export OMP_NUM_THREADS=1
OMP_OVERRIDE='+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS="1"'

"$VERL_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/prepare_sokoban_data.py" \
  --output-dir "$DATA_DIR"
bash "$PROJECT_DIR/scripts/run_verl_grpo_smoke.sh" "$VERL_DIR" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.experiment_name=qwen3.5-2b-sokoban-agent-smoke \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.rollout.prompt_length=1024 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=probegrpo_sokoban \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/configs/agent_loop/sokoban.yaml" \
  "$OMP_OVERRIDE" \
  "$@"
