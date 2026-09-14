#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
export PYTHONUNBUFFERED=1
.venv/bin/ai-trader run --config "${1:-configs/default.json}" --threads "${2:-0}" \
  2>&1 | tee "artifacts/logs/train-$(date -u +%Y%m%dT%H%M%SZ).log"
