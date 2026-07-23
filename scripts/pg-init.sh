#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the stocks conda environment first: conda activate stocks"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${CONDA_PREFIX}/var/lib/postgresql/data"
SCHEMA_FILE="${REPO_ROOT}/src/tradingai/data/schema.sql"

"${SCRIPT_DIR}/pg-start.sh"

init_database() {
  local db_name="$1"
  if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${db_name}'" | grep -q 1; then
    createdb "${db_name}"
    echo "Created database ${db_name}"
  fi
  psql -d "${db_name}" -f "${SCHEMA_FILE}"
  echo "Applied schema to ${db_name}"
}

if [[ -n "${TRADINGAI_DATABASE_NAME:-}" ]]; then
  init_database "${TRADINGAI_DATABASE_NAME}"
else
  init_database "tradingai"
  init_database "tradingai_dev"
fi
