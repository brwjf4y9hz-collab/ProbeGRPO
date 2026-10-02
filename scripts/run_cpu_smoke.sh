#!/usr/bin/env bash
# 本脚本执行 run cpu smoke 对应的仓库流程；具体参数、环境变量和命令保持原样。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m probegrpo.smoke
python3 -m probegrpo.agent_smoke
