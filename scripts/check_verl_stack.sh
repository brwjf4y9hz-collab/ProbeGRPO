#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/verl" >&2
  exit 2
fi

VERL_DIR="$1"
EXPECTED_COMMIT="${VERL_COMMIT:-cf14ded3a448107e70a206fd201817cc1cbae348}"

if [[ ! -f "$VERL_DIR/pyproject.toml" || ! -d "$VERL_DIR/verl" ]]; then
  echo "Not a verl checkout: $VERL_DIR" >&2
  exit 2
fi
if [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Unexpected verl revision. Expected $EXPECTED_COMMIT." >&2
  exit 1
fi
if [[ ! -x "$VERL_DIR/.venv/bin/python" ]]; then
  echo "Missing verl uv environment. Run scripts/bootstrap_verl.sh first." >&2
  exit 1
fi

cd "$VERL_DIR"
"$VERL_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata

import torch
import transformers
import vllm
from transformers import AutoConfig

import probegrpo
import verl

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"vllm={vllm.__version__}")
print(f"verl={importlib.metadata.version('verl')}")
print(f"probegrpo={probegrpo.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"gpu_memory_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}")

config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
print(f"model_type={config.model_type}")
if config.model_type != "qwen3_5":
    raise SystemExit(f"Unexpected Qwen3.5 model type: {config.model_type}")

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

if "extra_fields" not in AgentLoopOutput.model_fields:
    raise SystemExit("verl AgentLoopOutput no longer exposes extra_fields")
print("verl stack compatibility check passed")
PY
