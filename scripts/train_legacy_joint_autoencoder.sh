#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-worm}"
ROOT="${ROOT:-data/${DATASET}}"
CONFIG="${CONFIG:-configs/autoencoder/legacy-joint-kl-f4.yaml}"
RUN_NAME="${RUN_NAME:-${DATASET}-legacy-joint-kl-f4}"
GPUS="${GPUS:-0,}"
NO_TEST="${NO_TEST:-True}"

EXTRA_ARGS=(
  "data.params.train.params.root=${ROOT}"
  "data.params.validation.params.root=${ROOT}"
)

if [[ -n "${AE_INIT_CKPT:-}" ]]; then
  EXTRA_ARGS+=("model.params.ckpt_path=${AE_INIT_CKPT}")
fi

echo "DATASET=${DATASET}"
echo "ROOT=${ROOT}"
echo "CONFIG=${CONFIG}"
echo "RUN_NAME=${RUN_NAME}"
echo "GPUS=${GPUS}"
if [[ -n "${AE_INIT_CKPT:-}" ]]; then
  echo "AE_INIT_CKPT=${AE_INIT_CKPT}"
fi

python main.py \
  --base "$CONFIG" \
  -t \
  --name "$RUN_NAME" \
  --gpus "$GPUS" \
  --no-test "$NO_TEST" \
  "${EXTRA_ARGS[@]}" \
  "$@"
