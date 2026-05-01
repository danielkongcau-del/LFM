# Python 3.10 / CUDA 12.8 Install Guide

This project is adapted to run the worm paired-autoencoder workflow on a modern Python 3.10 environment. Install PyTorch first, then install the compatible runtime dependencies from this repository.

## Conda Environment

```bash
conda create -n ldm python=3.10 pip -y
conda activate ldm
```

## PyTorch

Use the CUDA 12.8 wheel index requested for this project:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Project Dependencies

```bash
pip install -r requirements/ldm-py310-cu128-minimal.txt
pip install -e .
```

Do not install the original `environment.yaml` as-is for this workflow. Its old pins include packages that either lack Python 3.10 wheels or pull obsolete build chains.
The minimal requirements include `lpips` for the image-only worm autoencoder and `taming-transformers` for original VQ first-stage models.

## Validation

Check that imports, CUDA, and dependency metadata are sane:

```bash
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Run a tiny CPU smoke test for the worm modality-token autoencoder:

```bash
python main.py --base configs/autoencoder/worm-paired-token-kl.yaml -t --no-test True --max_steps 1 --limit_val_batches 0 --num_sanity_val_steps 0 data.params.batch_size=1 data.params.train.params.size=64 data.params.validation.params.size=64 model.params.ddconfig.resolution=64 model.params.ddconfig.ch=32 model.params.ddconfig.ch_mult=[1,2,4] model.params.token_dim=32 model.params.latent_blocks=1
```

For GPU training, prefer the modern Lightning arguments:

```bash
python main.py --base configs/autoencoder/worm-paired-token-kl.yaml -t --accelerator gpu --devices 1 --no-test True
```

If you use the older Lightning `--gpus` form in PowerShell, quote comma-suffixed GPU ids, for example `--gpus "0,"`. A bare `--gpus 0,` can be parsed by PowerShell as `--gpus 0`, which means CPU execution.

## Dataset

The entire `data/` directory is intentionally ignored by git. Place the worm dataset at:

```text
data/worm/
  train/image/
  train/mask/
  validation or val/image/
  validation or val/mask/
  test/image/
  test/mask/
```

The committed configs currently use `data/worm` and the split names already present in the copied dataset.
