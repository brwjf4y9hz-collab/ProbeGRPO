#!/usr/bin/env bash
# 本脚本执行 smoke test 对应的仓库流程；具体参数、环境变量和命令保持原样。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m probegrpo.smoke

