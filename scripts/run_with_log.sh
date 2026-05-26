#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: scripts/run_with_log.sh <log-name> <command> [args...]" >&2
    exit 2
fi

LOG_NAME="$1"
shift

WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
mkdir -p "${LOG_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_NAME="$(printf '%s' "${LOG_NAME}" | tr -c '[:alnum:]_.-' '_')"
LOG_PATH="${LOG_DIR}/${STAMP}_${SAFE_NAME}.log"

{
    echo "======================================================================"
    echo "MITS logged command"
    echo "======================================================================"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Work dir: ${WORK_DIR}"
    echo "Log path: ${LOG_PATH}"
    echo "Command: $*"
    echo "======================================================================"
} | tee "${LOG_PATH}"

set +e
"$@" 2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e

{
    echo "======================================================================"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Exit code: ${STATUS}"
    echo "======================================================================"
} | tee -a "${LOG_PATH}"

exit "${STATUS}"
