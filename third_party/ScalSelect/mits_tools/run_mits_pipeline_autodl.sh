#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# AutoDL defaults. Override any value by exporting the same variable before running.
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data/dataset}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"

FEATURE_MODE="${FEATURE_MODE:-hybrid_meta}"      # hybrid_meta or qwen_attention
TEXT_ALPHA="${TEXT_ALPHA:-0.30}"
META_ALPHA="${META_ALPHA:-0.15}"
RATIO="${RATIO:-15}"
MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-32}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-8}"
QA_FILTER="${QA_FILTER:-balanced}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SV_THRESHOLD="${SV_THRESHOLD:-0.9}"
GROUP_BY="${GROUP_BY:-scene}"
MIN_PER_GROUP="${MIN_PER_GROUP:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_mits_pipeline_ratio${RATIO}.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

INDEX_PATH="${WORK_DIR}/mits_index.jsonl"
SHAREGPT_PATH="${WORK_DIR}/mits_sharegpt.json"
FEATURE_DIR="${WORK_DIR}/features"
SCORE_PATH="${WORK_DIR}/importance_scores.jsonl"
SELECTED_INDEX_PATH="${WORK_DIR}/mits_selected_${RATIO}.jsonl"
SELECTED_SHAREGPT_PATH="${WORK_DIR}/mits_selected_${RATIO}_sharegpt.json"
GROUP_SUMMARY_PATH="${WORK_DIR}/mits_selected_${RATIO}_group_summary.jsonl"

export DATASET_ROOT MODEL_PATH WORK_DIR
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "======================================================================"
echo "MITS AutoDL ScalSelect Pipeline"
echo "======================================================================"
echo "Dataset root: ${DATASET_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Work dir: ${WORK_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Pipeline log: ${PIPELINE_LOG}"
echo "Feature mode: ${FEATURE_MODE}"
echo "Selection ratio: ${RATIO}"
echo "QA filter: ${QA_FILTER}"
echo "Max pairs/sample: ${MAX_PAIRS_PER_SAMPLE}"
echo "Max pairs/task: ${MAX_PAIRS_PER_TASK}"
echo "======================================================================"

python third_party/ScalSelect/mits_tools/build_mits_index.py \
    --dataset-root "${DATASET_ROOT}" \
    --output "${INDEX_PATH}" \
    --limit 0 \
    --allow-full-scan

python third_party/ScalSelect/mits_tools/convert_mits_to_sharegpt.py \
    --dataset-root "${DATASET_ROOT}" \
    --index "${INDEX_PATH}" \
    --output "${SHAREGPT_PATH}" \
    --max-pairs-per-sample "${MAX_PAIRS_PER_SAMPLE}" \
    --qa-filter "${QA_FILTER}" \
    --max-pairs-per-task "${MAX_PAIRS_PER_TASK}"

if [ "${NUM_PROCESSES}" = "1" ]; then
    python third_party/ScalSelect/scripts/feature_extract_sft.py \
        --model "${MODEL_PATH}" \
        --model-type qwen \
        --dataset "${SHAREGPT_PATH}" \
        --output-dir "${FEATURE_DIR}" \
        --feature-mode "${FEATURE_MODE}" \
        --text-alpha "${TEXT_ALPHA}" \
        --meta-alpha "${META_ALPHA}" \
        --max-samples "${MAX_SAMPLES}" \
        --sample-batch-size "${SAMPLE_BATCH_SIZE}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --max-length "${MAX_LENGTH}"
else
    accelerate launch \
        --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 \
        --mixed_precision bf16 \
        third_party/ScalSelect/scripts/feature_extract_sft.py \
        --model "${MODEL_PATH}" \
        --model-type qwen \
        --dataset "${SHAREGPT_PATH}" \
        --output-dir "${FEATURE_DIR}" \
        --feature-mode "${FEATURE_MODE}" \
        --text-alpha "${TEXT_ALPHA}" \
        --meta-alpha "${META_ALPHA}" \
        --max-samples "${MAX_SAMPLES}" \
        --sample-batch-size "${SAMPLE_BATCH_SIZE}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --max-length "${MAX_LENGTH}"
fi

python third_party/ScalSelect/scripts/cur.py \
    --features-dir "${FEATURE_DIR}" \
    --output "${SCORE_PATH}" \
    --sv-threshold "${SV_THRESHOLD}"

python third_party/ScalSelect/mits_tools/select_mits_subset.py \
    --index "${INDEX_PATH}" \
    --cur-scores "${SCORE_PATH}" \
    --output "${SELECTED_INDEX_PATH}" \
    --ratio "${RATIO}" \
    --group-by "${GROUP_BY}" \
    --min-per-group "${MIN_PER_GROUP}" \
    --group-summary-output "${GROUP_SUMMARY_PATH}"

python third_party/ScalSelect/mits_tools/convert_mits_to_sharegpt.py \
    --dataset-root "${DATASET_ROOT}" \
    --index "${SELECTED_INDEX_PATH}" \
    --output "${SELECTED_SHAREGPT_PATH}" \
    --max-pairs-per-sample "${MAX_PAIRS_PER_SAMPLE}" \
    --qa-filter "${QA_FILTER}" \
    --max-pairs-per-task "${MAX_PAIRS_PER_TASK}"

echo "======================================================================"
echo "Pipeline complete."
echo "Full ShareGPT: ${SHAREGPT_PATH}"
echo "Features: ${FEATURE_DIR}/all_representations.npz"
echo "Scores: ${SCORE_PATH}"
echo "Selected index: ${SELECTED_INDEX_PATH}"
echo "Selected ShareGPT: ${SELECTED_SHAREGPT_PATH}"
echo "Group summary: ${GROUP_SUMMARY_PATH}"
echo "======================================================================"
