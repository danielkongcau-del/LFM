# Worm Mask Improved-Diffusion Baseline

This baseline trains OpenAI `improved-diffusion` directly on thresholded worm mask
PNGs. The masks are exported as RGB images because the upstream dataloader always
converts inputs to RGB.

Prepare data:

```bash
python scripts/improved_diffusion/prepare_worm_mask_dataset.py \
  --root data/worm \
  --out_root data/worm-improved-diffusion-mask256 \
  --splits train val \
  --size 256
```

Train one GPU:

```bash
OPENAI_LOGDIR=logs/improved-diffusion/worm-mask-256 \
DATA_DIR=data/worm-improved-diffusion-mask256/train \
BATCH_SIZE=4 \
MICROBATCH=1 \
SAVE_EVERY_EPOCHS=20 \
SAMPLE_EVERY_EPOCHS=100 \
KEEP_LATEST=True \
bash scripts/improved_diffusion/train_worm_mask256.sh
```

Train two GPUs:

```bash
OPENAI_LOGDIR=logs/improved-diffusion/worm-mask-256 \
DATA_DIR=data/worm-improved-diffusion-mask256/train \
NUM_GPUS=2 \
BATCH_SIZE=4 \
MICROBATCH=1 \
SAVE_EVERY_EPOCHS=20 \
SAMPLE_EVERY_EPOCHS=100 \
KEEP_LATEST=True \
bash scripts/improved_diffusion/train_worm_mask256.sh
```

With `KEEP_LATEST=True`, each save removes older `model*.pt`, `ema_*.pt`, and
`opt*.pt` files in `OPENAI_LOGDIR`.

The training wrapper counts images under `DATA_DIR` and converts epoch-based
settings to training steps. By default it saves about every 20 epochs and writes
automatic DDIM-20 samples about every 100 epochs to:

```bash
logs/improved-diffusion/worm-mask-256/samples
```

The automatic sample directory keeps only the latest batch when
`SAMPLE_KEEP_LATEST=True`.

Sample from an EMA checkpoint:

```bash
NUM_SAMPLES=64 \
BATCH_SIZE=8 \
TIMESTEP_RESPACING=ddim20 \
bash scripts/improved_diffusion/sample_worm_mask256.sh \
  logs/improved-diffusion/worm-mask-256/ema_0.9999_XXXXX.pt \
  logs/improved-diffusion/worm-mask-256-samples
```

Thresholded PNG masks are written to:

```bash
logs/improved-diffusion/worm-mask-256-samples/png
```

The sample wrapper removes older `samples_*.npz` files and the previous `png`
directory before writing the latest sample batch.
