#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/third_party/improved-diffusion:${PYTHONPATH:-}"
export OPENAI_LOGDIR="${OPENAI_LOGDIR:-logs/improved-diffusion/worm-mask-256}"

DATA_DIR="${DATA_DIR:-data/worm-improved-diffusion-mask256/train}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICROBATCH="${MICROBATCH:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LR="${LR:-1e-4}"
USE_FP16="${USE_FP16:-False}"

MODEL_FLAGS=(
  --image_size 256
  --num_channels 64
  --num_res_blocks 2
  --num_heads 1
  --attention_resolutions 16,8
  --learn_sigma True
  --use_checkpoint True
)

DIFFUSION_FLAGS=(
  --diffusion_steps 1000
  --noise_schedule cosine
)

TRAIN_FLAGS=(
  --data_dir "$DATA_DIR"
  --lr "$LR"
  --batch_size "$BATCH_SIZE"
  --microbatch "$MICROBATCH"
  --save_interval "$SAVE_INTERVAL"
  --log_interval "$LOG_INTERVAL"
  --use_fp16 "$USE_FP16"
)

echo "OPENAI_LOGDIR=$OPENAI_LOGDIR"
echo "DATA_DIR=$DATA_DIR"
echo "NUM_GPUS=$NUM_GPUS BATCH_SIZE_PER_RANK=$BATCH_SIZE MICROBATCH=$MICROBATCH"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  mpiexec -n "$NUM_GPUS" python third_party/improved-diffusion/scripts/image_train.py \
    "${MODEL_FLAGS[@]}" "${DIFFUSION_FLAGS[@]}" "${TRAIN_FLAGS[@]}"
else
  python third_party/improved-diffusion/scripts/image_train.py \
    "${MODEL_FLAGS[@]}" "${DIFFUSION_FLAGS[@]}" "${TRAIN_FLAGS[@]}"
fi
