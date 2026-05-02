#!/usr/bin/env bash
set -euo pipefail

CONFIGS="${CONFIGS:-configs/latent-diffusion/worm-legacy-joint-ldm-uncond-f4.yaml}"
CKPT="${CKPT:-}"
OUTDIR="${OUTDIR:-outputs/worm-legacy-joint-ldm-samples}"
N_SAMPLES="${N_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DDIM_STEPS="${DDIM_STEPS:-50}"
DDIM_ETA="${DDIM_ETA:-0.0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-23}"

read -r -a CONFIG_ARGS <<< "${CONFIGS}"

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
  EXTRA_ARGS+=("--ckpt" "${CKPT}")
fi

python scripts/sample_worm_legacy_joint_ldm.py \
  --config "${CONFIG_ARGS[@]}" \
  --outdir "${OUTDIR}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --ddim_steps "${DDIM_STEPS}" \
  --ddim_eta "${DDIM_ETA}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
