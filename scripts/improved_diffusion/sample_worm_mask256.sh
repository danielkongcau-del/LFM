#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/model_or_ema_checkpoint.pt [out_logdir]" >&2
  exit 2
fi

MODEL_PATH="$1"
export PYTHONPATH="$ROOT_DIR/third_party/improved-diffusion:${PYTHONPATH:-}"
export OPENAI_LOGDIR="${2:-logs/improved-diffusion/worm-mask-256-samples}"

NUM_SAMPLES="${NUM_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TIMESTEP_RESPACING="${TIMESTEP_RESPACING:-ddim250}"
USE_DDIM="${USE_DDIM:-True}"

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
  --timestep_respacing "$TIMESTEP_RESPACING"
)

SAMPLE_FLAGS=(
  --model_path "$MODEL_PATH"
  --num_samples "$NUM_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --use_ddim "$USE_DDIM"
)

echo "OPENAI_LOGDIR=$OPENAI_LOGDIR"
python third_party/improved-diffusion/scripts/image_sample.py \
  "${MODEL_FLAGS[@]}" "${DIFFUSION_FLAGS[@]}" "${SAMPLE_FLAGS[@]}"

NPZ_PATH="$(ls -t "$OPENAI_LOGDIR"/samples_*.npz | head -n 1)"
python scripts/improved_diffusion/npz_to_mask_pngs.py \
  --npz "$NPZ_PATH" \
  --outdir "$OPENAI_LOGDIR/png" \
  --save_raw

echo "Samples: $OPENAI_LOGDIR/png"
