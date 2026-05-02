import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldm.data.worm import WormPairsBase
from ldm.util import instantiate_from_config


def trusted_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(config_path, checkpoint_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    payload = trusted_torch_load(checkpoint_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded {} with {} missing and {} unexpected keys".format(
        checkpoint_path, len(missing), len(unexpected)
    ))
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, config


def batch_mask_to_device(batch, device):
    mask = batch["mask"]
    if mask.ndim == 3:
        mask = mask[..., None]
    if mask.shape[1] != 1:
        mask = mask.permute(0, 3, 1, 2)
    return mask.to(device=device, dtype=torch.float32, memory_format=torch.contiguous_format)


def safe_stem(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq_config", default="configs/autoencoder/worm-mask-vq.yaml")
    parser.add_argument("--vq_ckpt", required=True)
    parser.add_argument("--root", default="data/worm")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--max_samples", type=int)
    return parser.parse_args()


@torch.no_grad()
def main():
    opt = parse_args()
    if opt.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(opt.device)
    model, config = load_model(opt.vq_config, opt.vq_ckpt, device)
    dataset = WormPairsBase(
        root=opt.root,
        split=opt.split,
        size=opt.size,
        random_crop=False,
        flip_p=0.0,
        interpolation="bicubic",
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False,
    )

    os.makedirs(opt.outdir, exist_ok=True)
    count = 0
    n_embed = int(config.model.params.n_embed)
    embed_dim = int(config.model.params.embed_dim)

    for batch in tqdm(loader, desc="Caching VQ mask tokens"):
        mask = batch_mask_to_device(batch, device)
        indices = model.encode_to_indices(mask).reshape(mask.shape[0], -1)
        grid_size = int(round(indices.shape[1] ** 0.5))
        if grid_size * grid_size != indices.shape[1]:
            raise ValueError("Token count {} is not a square grid.".format(indices.shape[1]))

        if opt.no_flip:
            indices_flip = None
        else:
            indices_flip = model.encode_to_indices(mask.flip(dims=[3])).reshape(mask.shape[0], -1)

        names = batch["relative_file_path_"]
        for i, name in enumerate(names):
            if opt.max_samples is not None and count >= opt.max_samples:
                break

            stem = safe_stem(name)
            path = os.path.join(opt.outdir, "{:06d}_{}.npz".format(count, stem))
            data = {
                "idx": indices[i].detach().cpu().numpy().astype(np.int64),
                "grid_size": np.asarray(grid_size, dtype=np.int64),
                "n_embed": np.asarray(n_embed, dtype=np.int64),
                "embed_dim": np.asarray(embed_dim, dtype=np.int64),
                "source_name": np.asarray(name),
            }
            if indices_flip is not None:
                data["idx_flip"] = indices_flip[i].detach().cpu().numpy().astype(np.int64)
            np.savez(path, **data)
            count += 1

        if opt.max_samples is not None and count >= opt.max_samples:
            break

    print("Wrote {} token cache files to {}".format(count, opt.outdir))


if __name__ == "__main__":
    main()
