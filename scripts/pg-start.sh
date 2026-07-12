#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the stocks conda environment first: conda activate stocks"
  exit 1
fi

DATA_DIR="${CONDA_PREFIX}/var/lib/postgresql/data"
LOG_FILE="${HOME}/tradingai-pg.log"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Initializing PostgreSQL data directory at ${DATA_DIR}"
  initdb -D "${DATA_DIR}"
fi

if pg_ctl -D "${DATA_DIR}" status >/dev/null 2>&1; then
  echo "PostgreSQL is already running"
else
  pg_ctl -D "${DATA_DIR}" -l "${LOG_FILE}" start
  echo "PostgreSQL started. Log: ${LOG_FILE}"
fi
