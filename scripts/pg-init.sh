#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the stocks conda environment first: conda activate stocks"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${CONDA_PREFIX}/var/lib/postgresql/data"
DB_NAME="${TRADINGAI_DATABASE_NAME:-tradingai}"

"${SCRIPT_DIR}/pg-start.sh"

if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  createdb "${DB_NAME}"
  echo "Created database ${DB_NAME}"
fi

psql -d "${DB_NAME}" -f "${REPO_ROOT}/src/tradingai/data/schema.sql"
echo "Applied schema to ${DB_NAME}"
