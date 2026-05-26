#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data/dataset}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
TRAFFIC_MODEL="${TRAFFIC_MODEL:-/root/autodl-tmp/zhaokaikai/Qwen2.5-VL-7B-Instruct-Traffic}"
OURS_ADAPTER="${OURS_ADAPTER:-/root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_full_swanlab/v0-20260523-162150/checkpoint-1540}"
MERGED_OURS_MODEL="${MERGED_OURS_MODEL:-/root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_merged}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
TRAIN_DATASET="${TRAIN_DATASET:-${WORK_DIR}/mits_selected_15_train32_sharegpt.jsonl}"
MITS_INDEX_PATH="${MITS_INDEX_PATH:-${WORK_DIR}/mits_index.jsonl}"

EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/eval_test1000_merged}"
PRED_DIR="${PRED_DIR:-${EVAL_DIR}/predictions}"
SCORE_DIR="${SCORE_DIR:-${EVAL_DIR}/scores}"
COMPARE_DIR="${COMPARE_DIR:-${EVAL_DIR}/compare}"
SPEED_DIR="${SPEED_DIR:-${EVAL_DIR}/speed}"
ALLOW_FROZEN_EVAL_WRITE="${ALLOW_FROZEN_EVAL_WRITE:-0}"

VAL_IMAGES="${VAL_IMAGES:-0}"
TEST_IMAGES="${TEST_IMAGES:-1000}"
MAX_QAS_PER_IMAGE="${MAX_QAS_PER_IMAGE:-5}"
MAX_PIXELS="${MAX_PIXELS:-2073600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
PREDICT_PROGRESS_EVERY="${PREDICT_PROGRESS_EVERY:-1}"
PREDICT_RESUME="${PREDICT_RESUME:-1}"

AUTHOR_TEST_PATH="${AUTHOR_TEST_PATH:-}"
AUTHOR_TEST_CANDIDATES=(
    "${WORK_DIR}/swift_data/v1.0_test.jsonl"
    "${DATASET_ROOT}/swift_data/v1.0_test.jsonl"
    "/root/autodl-tmp/data/swift_data/v1.0_test.jsonl"
    "/root/autodl-tmp/data/dataset/swift_data/v1.0_test.jsonl"
)

export DATASET_ROOT MODEL_PATH WORK_DIR LOG_DIR
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [ "${ALLOW_FROZEN_EVAL_WRITE}" != "1" ]; then
    case "$(basename "${EVAL_DIR}")" in
        eval_test1000)
            echo "Refusing to write into frozen EVAL_DIR=${EVAL_DIR}. Use eval_test1000_merged/eval_speed_merged or set ALLOW_FROZEN_EVAL_WRITE=1 intentionally." >&2
            exit 1
            ;;
    esac
fi

mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${PRED_DIR}" "${SCORE_DIR}" "${COMPARE_DIR}" "${SPEED_DIR}"

EVAL_KIND="${EVAL_KIND:-auto}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

if [ "${EVAL_KIND}" = "auto" ]; then
    if [ -z "${AUTHOR_TEST_PATH}" ]; then
        for candidate in "${AUTHOR_TEST_CANDIDATES[@]}"; do
            if [ -s "${candidate}" ]; then
                AUTHOR_TEST_PATH="${candidate}"
                break
            fi
        done
    fi
    if [ -n "${AUTHOR_TEST_PATH}" ] && [ -s "${AUTHOR_TEST_PATH}" ]; then
        EVAL_KIND="official"
    else
        EVAL_KIND="mits"
    fi
fi

if [ "${EVAL_KIND}" = "official" ]; then
    if [ -z "${AUTHOR_TEST_PATH}" ] || [ ! -s "${AUTHOR_TEST_PATH}" ]; then
        echo "AUTHOR_TEST_PATH is required for EVAL_KIND=official and must point to v1.0_test.jsonl." >&2
        exit 1
    fi
    EVAL_JSONL="${EVAL_JSONL:-${EVAL_DIR}/official_v1_test_eval.jsonl}"
    if [ "${REBUILD_EVAL_SET:-0}" = "1" ] || [ ! -s "${EVAL_JSONL}" ]; then
        bash scripts/run_with_log.sh "${LOG_NAME_CONVERT_OFFICIAL:-convert_official_v1_test_eval_merged}" \
            python third_party/ScalSelect/mits_tools/convert_external_vqa_to_eval.py \
                --input "${AUTHOR_TEST_PATH}" \
                --output "${EVAL_JSONL}" \
                --image-root "${DATASET_ROOT}/images" \
                --scene official_test \
                --source-name v1.0_test
    fi
elif [ "${EVAL_KIND}" = "mits" ]; then
    EVAL_JSONL="${EVAL_JSONL:-${EVAL_DIR}/mits_${EVAL_SPLIT}_qas.jsonl}"
    if [ "${REBUILD_EVAL_SET:-0}" = "1" ] || [ ! -s "${EVAL_JSONL}" ]; then
        bash scripts/run_with_log.sh "${LOG_NAME_BUILD:-build_mits_eval_set_merged}" \
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
else
    if [ -z "${EVAL_JSONL:-}" ] || [ ! -s "${EVAL_JSONL}" ]; then
        echo "For EVAL_KIND=${EVAL_KIND}, set EVAL_JSONL to an existing eval JSONL." >&2
        exit 1
    fi
fi

LIMIT_ARGS=()
LIMIT_LABEL=""
if [ -n "${EVAL_LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
    LIMIT_LABEL="_limit${EVAL_LIMIT}"
fi

RESUME_ARGS=()
if [ "${PREDICT_RESUME}" = "1" ]; then
    RESUME_ARGS=(--resume)
fi

RUN_LABEL="${EVAL_KIND}_${EVAL_SPLIT}${LIMIT_LABEL}"
BASE_PRED="${PRED_DIR}/base_${RUN_LABEL}.jsonl"
TRAFFIC_PRED="${PRED_DIR}/traffic_full_${RUN_LABEL}.jsonl"
OURS_ADAPTER_PRED="${PRED_DIR}/ours_lorasculpt_adapter_${RUN_LABEL}.jsonl"
OURS_MERGED_PRED="${PRED_DIR}/ours_lorasculpt_merged_${RUN_LABEL}.jsonl"
BASE_SCORE_DIR="${SCORE_DIR}/base_${RUN_LABEL}"
TRAFFIC_SCORE_DIR="${SCORE_DIR}/traffic_full_${RUN_LABEL}"
OURS_ADAPTER_SCORE_DIR="${SCORE_DIR}/ours_lorasculpt_adapter_${RUN_LABEL}"
OURS_MERGED_SCORE_DIR="${SCORE_DIR}/ours_lorasculpt_merged_${RUN_LABEL}"
EXTERNAL_SCORE_ARGS=()
if [ "${EVAL_KIND}" != "mits" ]; then
    EXTERNAL_SCORE_ARGS=(--external)
fi

run_predict() {
    local label="$1"
    local model_path="$2"
    local output_path="$3"
    shift 3

    bash scripts/run_with_log.sh "eval_${label}_${RUN_LABEL}" \
        python third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py \
            --eval-jsonl "${EVAL_JSONL}" \
            --output "${output_path}" \
            --model "${model_path}" \
            --model-label "${label}" \
            --device-map "${DEVICE_MAP}" \
            --torch-dtype "${TORCH_DTYPE}" \
            --max-pixels "${MAX_PIXELS}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --progress-every "${PREDICT_PROGRESS_EVERY}" \
            "${LIMIT_ARGS[@]}" \
            "${RESUME_ARGS[@]}" \
            "$@"
}

run_score() {
    local label="$1"
    local pred_path="$2"
    local out_dir="$3"
    bash scripts/run_with_log.sh "score_${label}_${RUN_LABEL}" \
        python third_party/ScalSelect/mits_tools/score_mits_predictions.py \
            --predictions "${pred_path}" \
            --output-dir "${out_dir}" \
            "${EXTERNAL_SCORE_ARGS[@]}"
}

run_predict base "${MODEL_PATH}" "${BASE_PRED}"
run_predict traffic_full "${TRAFFIC_MODEL}" "${TRAFFIC_PRED}"
run_predict ours_15_lorasculpt_adapter "${MODEL_PATH}" "${OURS_ADAPTER_PRED}" --adapter "${OURS_ADAPTER}"
run_predict ours_15_lorasculpt_merged "${MERGED_OURS_MODEL}" "${OURS_MERGED_PRED}"

BASE_LINES="$(wc -l < "${BASE_PRED}")"
TRAFFIC_LINES="$(wc -l < "${TRAFFIC_PRED}")"
OURS_ADAPTER_LINES="$(wc -l < "${OURS_ADAPTER_PRED}")"
OURS_MERGED_LINES="$(wc -l < "${OURS_MERGED_PRED}")"
if [ "${BASE_LINES}" != "${TRAFFIC_LINES}" ] || [ "${BASE_LINES}" != "${OURS_ADAPTER_LINES}" ] || [ "${BASE_LINES}" != "${OURS_MERGED_LINES}" ]; then
    echo "Prediction row count mismatch: base=${BASE_LINES}, traffic=${TRAFFIC_LINES}, ours_adapter=${OURS_ADAPTER_LINES}, ours_merged=${OURS_MERGED_LINES}" >&2
    exit 1
fi

run_score base "${BASE_PRED}" "${BASE_SCORE_DIR}"
run_score traffic_full "${TRAFFIC_PRED}" "${TRAFFIC_SCORE_DIR}"
run_score ours_15_lorasculpt_adapter "${OURS_ADAPTER_PRED}" "${OURS_ADAPTER_SCORE_DIR}"
run_score ours_15_lorasculpt_merged "${OURS_MERGED_PRED}" "${OURS_MERGED_SCORE_DIR}"

bash scripts/run_with_log.sh "compare_base_traffic_ours_merged_${RUN_LABEL}" \
    python third_party/ScalSelect/mits_tools/compare_mits_eval_multi.py \
        --summary "base=${BASE_SCORE_DIR}/summary.json" \
        --summary "traffic_full=${TRAFFIC_SCORE_DIR}/summary.json" \
        --summary "ours_15_lorasculpt_adapter=${OURS_ADAPTER_SCORE_DIR}/summary.json" \
        --summary "ours_15_lorasculpt_merged=${OURS_MERGED_SCORE_DIR}/summary.json" \
        --ours-label "ours_15_lorasculpt_merged" \
        --output-md "${COMPARE_DIR}/base_vs_traffic_vs_ours_merged_${RUN_LABEL}.md" \
        --output-csv "${COMPARE_DIR}/base_vs_traffic_vs_ours_merged_${RUN_LABEL}.csv"

bash scripts/run_with_log.sh "speed_base_traffic_ours_merged_${RUN_LABEL}" \
    python third_party/ScalSelect/mits_tools/summarize_prediction_speed.py \
        --prediction "base=${BASE_PRED}" \
        --prediction "traffic_full=${TRAFFIC_PRED}" \
        --prediction "ours_15_lorasculpt_adapter=${OURS_ADAPTER_PRED}" \
        --prediction "ours_15_lorasculpt_merged=${OURS_MERGED_PRED}" \
        --output-json "${SPEED_DIR}/prediction_speed_${RUN_LABEL}.json" \
        --output-csv "${SPEED_DIR}/prediction_speed_${RUN_LABEL}.csv" \
        --output-md "${SPEED_DIR}/prediction_speed_${RUN_LABEL}.md"

echo "Eval kind: ${EVAL_KIND}"
echo "Eval JSONL: ${EVAL_JSONL}"
echo "Base predictions: ${BASE_PRED}"
echo "Traffic predictions: ${TRAFFIC_PRED}"
echo "Ours adapter predictions: ${OURS_ADAPTER_PRED}"
echo "Ours merged predictions: ${OURS_MERGED_PRED}"
echo "Comparison: ${COMPARE_DIR}/base_vs_traffic_vs_ours_merged_${RUN_LABEL}.md"
echo "Speed summary: ${SPEED_DIR}/prediction_speed_${RUN_LABEL}.md"
