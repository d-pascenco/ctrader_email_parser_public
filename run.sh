#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  if grep -q $'\r' .env; then
    tmp_env="$(mktemp)"
    tr -d '\r' < .env > "$tmp_env"
    # shellcheck disable=SC1091
    source "$tmp_env"
    rm -f "$tmp_env"
  else
    # shellcheck disable=SC1091
    source .env
  fi
  set +a
fi

: "${PYTHON:=python3}"

# Ensure log file exists before appending
touch log.txt

exec "$PYTHON" ctrader.py >> log.txt 2>&1
