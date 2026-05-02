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
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-20}"
SAMPLE_EVERY_EPOCHS="${SAMPLE_EVERY_EPOCHS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LR="${LR:-1e-4}"
USE_FP16="${USE_FP16:-False}"
KEEP_LATEST="${KEEP_LATEST:-True}"
AUTO_SAMPLE="${AUTO_SAMPLE:-True}"
SAMPLE_NUM_SAMPLES="${SAMPLE_NUM_SAMPLES:-16}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SAMPLE_TIMESTEP_RESPACING="${SAMPLE_TIMESTEP_RESPACING:-ddim20}"
SAMPLE_USE_DDIM="${SAMPLE_USE_DDIM:-True}"
SAMPLE_KEEP_LATEST="${SAMPLE_KEEP_LATEST:-True}"
SAMPLE_SAVE_RAW="${SAMPLE_SAVE_RAW:-True}"
SAMPLE_DIR="${SAMPLE_DIR:-$OPENAI_LOGDIR/samples}"

DATASET_SIZE="${DATASET_SIZE:-$(python - "$DATA_DIR" <<'PY'
import os
import sys

root = sys.argv[1]
extensions = {".jpg", ".jpeg", ".png", ".gif"}
count = 0
for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
        if os.path.splitext(filename)[1].lower() in extensions:
            count += 1
print(count)
PY
)}"
if [[ "$DATASET_SIZE" -le 0 ]]; then
  echo "No training images found under DATA_DIR=$DATA_DIR" >&2
  exit 1
fi

GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS))
STEPS_PER_EPOCH=$((DATASET_SIZE / GLOBAL_BATCH))
if [[ "$STEPS_PER_EPOCH" -lt 1 ]]; then
  STEPS_PER_EPOCH=1
fi
if [[ -z "$SAVE_INTERVAL" ]]; then
  SAVE_INTERVAL=$((STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS))
fi
if [[ -z "$SAMPLE_INTERVAL" ]]; then
  SAMPLE_INTERVAL=$((STEPS_PER_EPOCH * SAMPLE_EVERY_EPOCHS))
fi

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
  --keep_latest "$KEEP_LATEST"
)

if [[ "$AUTO_SAMPLE" == "True" || "$AUTO_SAMPLE" == "true" || "$AUTO_SAMPLE" == "1" ]]; then
  TRAIN_FLAGS+=(
    --sample_interval "$SAMPLE_INTERVAL"
    --sample_num_samples "$SAMPLE_NUM_SAMPLES"
    --sample_batch_size "$SAMPLE_BATCH_SIZE"
    --sample_timestep_respacing "$SAMPLE_TIMESTEP_RESPACING"
    --sample_use_ddim "$SAMPLE_USE_DDIM"
    --sample_dir "$SAMPLE_DIR"
    --sample_keep_latest "$SAMPLE_KEEP_LATEST"
    --sample_save_raw "$SAMPLE_SAVE_RAW"
  )
fi

echo "OPENAI_LOGDIR=$OPENAI_LOGDIR"
echo "DATA_DIR=$DATA_DIR"
echo "NUM_GPUS=$NUM_GPUS BATCH_SIZE_PER_RANK=$BATCH_SIZE MICROBATCH=$MICROBATCH"
echo "DATASET_SIZE=$DATASET_SIZE GLOBAL_BATCH=$GLOBAL_BATCH STEPS_PER_EPOCH~=$STEPS_PER_EPOCH"
echo "SAVE_INTERVAL=$SAVE_INTERVAL (~${SAVE_EVERY_EPOCHS} epochs) KEEP_LATEST=$KEEP_LATEST"
echo "AUTO_SAMPLE=$AUTO_SAMPLE SAMPLE_INTERVAL=$SAMPLE_INTERVAL (~${SAMPLE_EVERY_EPOCHS} epochs) SAMPLE_TIMESTEP_RESPACING=$SAMPLE_TIMESTEP_RESPACING"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  mpiexec -n "$NUM_GPUS" python third_party/improved-diffusion/scripts/image_train.py \
    "${MODEL_FLAGS[@]}" "${DIFFUSION_FLAGS[@]}" "${TRAIN_FLAGS[@]}"
else
  python third_party/improved-diffusion/scripts/image_train.py \
    "${MODEL_FLAGS[@]}" "${DIFFUSION_FLAGS[@]}" "${TRAIN_FLAGS[@]}"
fi
