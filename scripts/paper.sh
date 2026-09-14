#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
AI_TRADER_RUN=$(.venv/bin/python -c 'import json; print(json.load(open("artifacts/latest.json"))["run_dir"])')
exec .venv/bin/ai-trader paper --run "$AI_TRADER_RUN" "$@"
