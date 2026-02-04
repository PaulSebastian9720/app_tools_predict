#!/usr/bin/env bash
# Launch the unified MLflow UI.
#
# Both notebooks/ and api_tols/ log to the same file-based store
# at notebooks/mlruns/.  This script starts the MLflow UI pointing there.
#
# Usage:
#   ./launch_mlflow_ui.sh          # default port 5000
#   ./launch_mlflow_ui.sh 5001     # custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLRUNS_DIR="${SCRIPT_DIR}/notebooks/mlruns"
PORT="${1:-5000}"

# Find the mlflow binary in either venv
if [ -x "${SCRIPT_DIR}/notebooks/venv/bin/mlflow" ]; then
    MLFLOW_BIN="${SCRIPT_DIR}/notebooks/venv/bin/mlflow"
elif [ -x "${SCRIPT_DIR}/api_tols/venv/bin/mlflow" ]; then
    MLFLOW_BIN="${SCRIPT_DIR}/api_tols/venv/bin/mlflow"
else
    echo "ERROR: mlflow not found in either venv." >&2
    echo "Install it with: pip install mlflow" >&2
    exit 1
fi

echo "MLflow UI"
echo "  Store : file://${MLRUNS_DIR}"
echo "  Port  : ${PORT}"
echo ""
echo "Open http://127.0.0.1:${PORT}"
echo "Ctrl+C to stop."
echo ""

exec "${MLFLOW_BIN}" ui \
    --backend-store-uri "file://${MLRUNS_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}"
