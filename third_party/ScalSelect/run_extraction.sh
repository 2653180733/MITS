#!/bin/bash

# Multi-turn Conversation Feature Extraction Script
# Extracts qwen_attention baseline or hybrid_meta sample representations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# NCCL Settings - prevent timeout on variable-length conversations
export NCCL_TIMEOUT=1800              # 30 minutes (default: 600s)
export NCCL_DEBUG=WARN                # Show warnings for debugging
export TORCH_NCCL_BLOCKING_WAIT=1     # Better error messages (new PyTorch API)
export NCCL_IB_TIMEOUT=22             # InfiniBand timeout

# 配置参数
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/data/outputs/full}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
MODEL="${MODEL_PATH:-/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET="${MITS_SHAREGPT_PATH:-${WORK_DIR}/mits_sharegpt.json}"
OUTPUT_DIR="${MITS_FEATURE_DIR:-${WORK_DIR}/features}"
FEATURE_MODE="${FEATURE_MODE:-hybrid_meta}"  # hybrid_meta or qwen_attention
TEXT_ALPHA="${TEXT_ALPHA:-0.30}"
META_ALPHA="${META_ALPHA:-0.15}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"  # -1 = all samples
NUM_PROCESSES="${NUM_PROCESSES:-1}"  # Number of GPUs
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"  # Number of samples to process together per device
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CUMULATIVE_THRESHOLD="${CUMULATIVE_THRESHOLD:-0.9}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_feature_extract.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "======================================================================"
echo "Multi-turn Conversation Feature Extraction"
echo "======================================================================"
echo "Log: ${LOG_PATH}"
echo "Model: ${MODEL}"
echo "Model type: ${MODEL_TYPE}"
echo "Feature mode: ${FEATURE_MODE}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Sample batch size: ${SAMPLE_BATCH_SIZE}"
echo "GPUs: ${NUM_PROCESSES}"
echo "Max length: ${MAX_LENGTH}"
echo "======================================================================"

# Run extraction
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=${NUM_PROCESSES} \
    --num_machines=1 \
    --mixed_precision=bf16 \
    scripts/feature_extract_sft.py \
    --model "${MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --feature-mode "${FEATURE_MODE}" \
    --text-alpha "${TEXT_ALPHA}" \
    --meta-alpha "${META_ALPHA}" \
    --max-samples ${MAX_SAMPLES} \
    --sample-batch-size ${SAMPLE_BATCH_SIZE} \
    --torch-dtype "${TORCH_DTYPE}" \
    --max-length ${MAX_LENGTH} \
    --cumulative-threshold ${CUMULATIVE_THRESHOLD}

echo ""
echo "======================================================================"
echo "✓ Feature extraction complete!"
echo "Vision representations: ${OUTPUT_DIR}/all_representations.npz"
echo "======================================================================"
