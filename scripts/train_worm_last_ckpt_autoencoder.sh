#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/autoencoder/worm-last-ckpt-joint-kl-f4.yaml}"
RUN_NAME="${RUN_NAME:-worm-last-ckpt-joint-kl-f4-ft}"
GPUS="${GPUS:-0,}"

python main.py \
  --base "$CONFIG" \
  -t \
  --name "$RUN_NAME" \
  --gpus "$GPUS"
