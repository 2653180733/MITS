#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data/dataset}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
RATIO="${RATIO:-15}"

TRAIN_DATASET="${TRAIN_DATASET:-${WORK_DIR}/mits_selected_${RATIO}_train32_sharegpt.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-/root/autodl-tmp/data/train_outputs/mits_${RATIO}_lorasculpt}"

export DATASET_ROOT MODEL_PATH WORK_DIR LOG_DIR
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export LORASCULPT_INTERVAL="${LORASCULPT_INTERVAL:-300}"
export LORASCULPT_PRESERVE_RATIO="${LORASCULPT_PRESERVE_RATIO:-0.10}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUT}"

REPORT_TO="${REPORT_TO:-tensorboard}"
read -r -a REPORT_TO_ARGS <<< "${REPORT_TO}"

SWIFT_ARGS=(
    swift sft
    --external_plugins scripts/swift_lorasculpt_plugin.py
    --model "${MODEL_PATH}"
    --tuner_type "${TUNER_TYPE:-lora}"
    --dataset "${TRAIN_DATASET}"
    --torch_dtype bfloat16
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
    --lora_rank "${LORA_RANK:-16}"
    --lora_alpha "${LORA_ALPHA:-32}"
    --learning_rate "${LEARNING_RATE:-1e-4}"
    --max_length "${MAX_LENGTH:-4096}"
    --save_steps "${SAVE_STEPS:-300}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-5}"
    --logging_steps "${LOGGING_STEPS:-5}"
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --output_dir "${TRAIN_OUT}"
    --attn_impl "${ATTN_IMPL:-sdpa}"
    --max_pixels "${MAX_PIXELS:-1048576}"
    --report_to "${REPORT_TO_ARGS[@]}"
)

if [ -n "${MAX_STEPS:-}" ]; then
    SWIFT_ARGS+=(--max_steps "${MAX_STEPS}")
fi

if [ "${ENABLE_SWANLAB:-0}" = "1" ]; then
    python -c "import swanlab" >/dev/null 2>&1 || {
        echo "SwanLab is not installed. Install it with: pip install swanlab -U" >&2
        exit 1
    }
    SWIFT_ARGS+=(
        --swanlab_project "${SWANLAB_PROJECT:-MITS-Qwen25VL}"
        --swanlab_exp_name "${SWANLAB_EXP_NAME:-mits_${RATIO}_lorasculpt_train32}"
        --swanlab_mode "${SWANLAB_MODE:-cloud}"
    )
fi

bash scripts/run_with_log.sh "${LOG_NAME:-train_swift_lorasculpt_ratio${RATIO}}" "${SWIFT_ARGS[@]}"
