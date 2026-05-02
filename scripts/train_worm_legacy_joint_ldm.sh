#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-configs/latent-diffusion/worm-legacy-joint-ldm-uncond-f4.yaml}"
WARMSTART_CONFIG="${WARMSTART_CONFIG:-configs/latent-diffusion/worm-legacy-joint-ldm-uncond-f4-warmstart.yaml}"
WARMSTART="${WARMSTART:-true}"
GPUS="${GPUS:-0,}"
NO_TEST="${NO_TEST:-True}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${WARMSTART}" == "true" || "${WARMSTART}" == "True" || "${WARMSTART}" == "1" ]]; then
    RUN_NAME="worm-legacy-joint-ldm-uncond-f4-warmstart"
  else
    RUN_NAME="worm-legacy-joint-ldm-uncond-f4-scratch"
  fi
fi

CONFIG_ARGS=("$BASE_CONFIG")
if [[ "${WARMSTART}" == "true" || "${WARMSTART}" == "True" || "${WARMSTART}" == "1" ]]; then
  CONFIG_ARGS+=("$WARMSTART_CONFIG")
fi

EXTRA_ARGS=()
if [[ -n "${WARMSTART_CKPT:-}" ]]; then
  EXTRA_ARGS+=("model.params.ckpt_path=${WARMSTART_CKPT}")
fi
if [[ -n "${AE_CKPT:-}" ]]; then
  EXTRA_ARGS+=("model.params.first_stage_config.params.ckpt_path=${AE_CKPT}")
fi

echo "BASE_CONFIG=${BASE_CONFIG}"
echo "WARMSTART=${WARMSTART}"
if [[ "${WARMSTART}" == "true" || "${WARMSTART}" == "True" || "${WARMSTART}" == "1" ]]; then
  echo "WARMSTART_CONFIG=${WARMSTART_CONFIG}"
  echo "WARMSTART_CKPT=${WARMSTART_CKPT:-logs/generator/original/last.ckpt}"
fi
echo "RUN_NAME=${RUN_NAME}"
echo "GPUS=${GPUS}"

python main.py \
  --base "${CONFIG_ARGS[@]}" \
  -t \
  --name "${RUN_NAME}" \
  --gpus "${GPUS}" \
  --no-test "${NO_TEST}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
