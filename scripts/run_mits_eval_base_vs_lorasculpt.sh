#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data/dataset}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
OURS_ADAPTER="${OURS_ADAPTER:-/root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_full_swanlab/v0-20260523-162150/checkpoint-1540}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
TRAIN_DATASET="${TRAIN_DATASET:-${WORK_DIR}/mits_selected_15_train32_sharegpt.jsonl}"
MITS_INDEX_PATH="${MITS_INDEX_PATH:-${WORK_DIR}/mits_index.jsonl}"

EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/eval}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_JSONL="${EVAL_JSONL:-${EVAL_DIR}/mits_${EVAL_SPLIT}_qas.jsonl}"
PRED_DIR="${PRED_DIR:-${EVAL_DIR}/predictions}"
SCORE_DIR="${SCORE_DIR:-${EVAL_DIR}/scores}"
COMPARE_DIR="${COMPARE_DIR:-${EVAL_DIR}/compare}"

VAL_IMAGES="${VAL_IMAGES:-1000}"
TEST_IMAGES="${TEST_IMAGES:-5000}"
MAX_QAS_PER_IMAGE="${MAX_QAS_PER_IMAGE:-5}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

export DATASET_ROOT MODEL_PATH WORK_DIR LOG_DIR
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${PRED_DIR}" "${SCORE_DIR}" "${COMPARE_DIR}"

if [ "${REBUILD_EVAL_SET:-0}" = "1" ] || [ ! -s "${EVAL_JSONL}" ]; then
    bash scripts/run_with_log.sh "${LOG_NAME_BUILD:-build_mits_eval_set}" \
        python third_party/ScalSelect/mits_tools/build_mits_eval_set.py \
            --dataset-root "${DATASET_ROOT}" \
            --index "${MITS_INDEX_PATH}" \
            --train-dataset "${TRAIN_DATASET}" \
            --output-dir "${EVAL_DIR}" \
            --val-images "${VAL_IMAGES}" \
            --test-images "${TEST_IMAGES}" \
            --max-qas-per-image "${MAX_QAS_PER_IMAGE}" \
            --require-train-dataset
fi

LIMIT_ARGS=()
LIMIT_LABEL=""
if [ -n "${EVAL_LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
    LIMIT_LABEL="_limit${EVAL_LIMIT}"
fi

BASE_PRED="${PRED_DIR}/base_${EVAL_SPLIT}${LIMIT_LABEL}.jsonl"
OURS_PRED="${PRED_DIR}/ours_lorasculpt_${EVAL_SPLIT}${LIMIT_LABEL}.jsonl"
BASE_SCORE_DIR="${SCORE_DIR}/base_${EVAL_SPLIT}${LIMIT_LABEL}"
OURS_SCORE_DIR="${SCORE_DIR}/ours_lorasculpt_${EVAL_SPLIT}${LIMIT_LABEL}"

bash scripts/run_with_log.sh "${LOG_NAME_BASE:-eval_base_qwen25vl_${EVAL_SPLIT}${LIMIT_LABEL}}" \
    python third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py \
        --eval-jsonl "${EVAL_JSONL}" \
        --output "${BASE_PRED}" \
        --model "${MODEL_PATH}" \
        --model-label base \
        --device-map "${DEVICE_MAP}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --max-pixels "${MAX_PIXELS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        ${LIMIT_ARGS[@]}

bash scripts/run_with_log.sh "${LOG_NAME_OURS:-eval_ours_lorasculpt_${EVAL_SPLIT}${LIMIT_LABEL}}" \
    python third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py \
        --eval-jsonl "${EVAL_JSONL}" \
        --output "${OURS_PRED}" \
        --model "${MODEL_PATH}" \
        --adapter "${OURS_ADAPTER}" \
        --model-label ours_15_lorasculpt \
        --device-map "${DEVICE_MAP}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --max-pixels "${MAX_PIXELS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        ${LIMIT_ARGS[@]}

BASE_LINES="$(wc -l < "${BASE_PRED}")"
OURS_LINES="$(wc -l < "${OURS_PRED}")"
if [ "${BASE_LINES}" != "${OURS_LINES}" ]; then
    echo "Prediction row count mismatch: base=${BASE_LINES}, ours=${OURS_LINES}" >&2
    exit 1
fi

bash scripts/run_with_log.sh "${LOG_NAME_SCORE_BASE:-score_base_${EVAL_SPLIT}${LIMIT_LABEL}}" \
    python third_party/ScalSelect/mits_tools/score_mits_predictions.py \
        --predictions "${BASE_PRED}" \
        --output-dir "${BASE_SCORE_DIR}"

bash scripts/run_with_log.sh "${LOG_NAME_SCORE_OURS:-score_ours_${EVAL_SPLIT}${LIMIT_LABEL}}" \
    python third_party/ScalSelect/mits_tools/score_mits_predictions.py \
        --predictions "${OURS_PRED}" \
        --output-dir "${OURS_SCORE_DIR}"

bash scripts/run_with_log.sh "${LOG_NAME_COMPARE:-compare_base_vs_ours_${EVAL_SPLIT}${LIMIT_LABEL}}" \
    python third_party/ScalSelect/mits_tools/compare_mits_eval.py \
        --base-summary "${BASE_SCORE_DIR}/summary.json" \
        --ours-summary "${OURS_SCORE_DIR}/summary.json" \
        --output-md "${COMPARE_DIR}/base_vs_ours_mits_eval${LIMIT_LABEL}.md" \
        --output-csv "${COMPARE_DIR}/base_vs_ours_mits_eval${LIMIT_LABEL}.csv"

echo "Eval JSONL: ${EVAL_JSONL}"
echo "Base predictions: ${BASE_PRED}"
echo "Ours predictions: ${OURS_PRED}"
echo "Base score summary: ${BASE_SCORE_DIR}/summary.json"
echo "Ours score summary: ${OURS_SCORE_DIR}/summary.json"
echo "Comparison: ${COMPARE_DIR}/base_vs_ours_mits_eval${LIMIT_LABEL}.md"
