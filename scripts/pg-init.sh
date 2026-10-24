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
SIM_SCHEMA_FILE="${REPO_ROOT}/src/tradingai/sim/schema.sql"

"${SCRIPT_DIR}/pg-start.sh"

init_database() {
  local db_name="$1"
  local schema_file="${2:-${SCHEMA_FILE}}"
  if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${db_name}'" | grep -q 1; then
    createdb "${db_name}"
    echo "Created database ${db_name}"
  fi
  psql -d "${db_name}" -f "${schema_file}"
  echo "Applied schema to ${db_name}"
}

if [[ -n "${TRADINGAI_DATABASE_NAME:-}" ]]; then
  if [[ "${TRADINGAI_DATABASE_NAME}" == "tradingai_sim" ]]; then
    init_database "tradingai_sim" "${SIM_SCHEMA_FILE}"
  else
    init_database "${TRADINGAI_DATABASE_NAME}"
  fi
else
  init_database "tradingai"
  init_database "tradingai_dev"
  init_database "tradingai_sim" "${SIM_SCHEMA_FILE}"
fi
