#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
RATIO="${RATIO:-15}"
TRAIN_DATASET="${TRAIN_DATASET:-${WORK_DIR}/mits_selected_${RATIO}_train64_sharegpt.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-/root/autodl-tmp/data/train_outputs/mits_${RATIO}_lora}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUT}"
export MODEL_PATH WORK_DIR LOG_DIR

bash scripts/run_with_log.sh "train_swift_ratio${RATIO}" \
    swift sft \
        --model "${MODEL_PATH}" \
        --tuner_type "${TUNER_TYPE:-lora}" \
        --dataset "${TRAIN_DATASET}" \
        --torch_dtype bfloat16 \
        --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
        --lora_rank "${LORA_RANK:-16}" \
        --lora_alpha "${LORA_ALPHA:-32}" \
        --learning_rate "${LEARNING_RATE:-1e-4}" \
        --max_length "${MAX_LENGTH:-4096}" \
        --save_steps "${SAVE_STEPS:-1000}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
        --logging_steps "${LOGGING_STEPS:-5}" \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --output_dir "${TRAIN_OUT}" \
        --attn_impl "${ATTN_IMPL:-flash_attn}" \
        --max_pixels "${MAX_PIXELS:-2073600}"
