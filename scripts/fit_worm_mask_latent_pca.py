import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent-diffusion/worm-mask-ldm-kl-f4.yaml")
    parser.add_argument("--out", default="logs/latent-diffusion/worm-mask-latent-pca/prior.pt")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main():
    opt = parse_args()
    if opt.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    first_stage = instantiate_from_config(config.model.params.first_stage_config)
    first_stage.to(device)
    first_stage.eval()

    dataset_cfg = config.data.params[opt.split]
    dataset = instantiate_from_config(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False,
    )

    latents = []
    latent_shape = None
    for batch_idx, batch in enumerate(loader):
        mask = first_stage.batch_to_mask(batch, device=device)
        z = first_stage.encode(mask).detach().float().cpu()
        latent_shape = tuple(z.shape[1:])
        latents.append(z.flatten(1))
        print("encoded batch {} / {} latents".format(batch_idx, len(loader)))

    x = torch.cat(latents, dim=0)
    if x.shape[0] < 2:
        raise ValueError("Need at least two masks to fit PCA.")

    n_components = min(opt.n_components, x.shape[0] - 1, x.shape[1])
    mean = x.mean(dim=0)
    centered = x - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].contiguous()
    scores = centered @ components.t()
    score_std = scores.std(dim=0, unbiased=True).clamp_min(1.0e-6)

    payload = {
        "mean": mean,
        "components": components,
        "scores": scores,
        "score_std": score_std,
        "latent_shape": latent_shape,
        "source_config": opt.config,
        "split": opt.split,
    }

    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    torch.save(payload, opt.out)
    print("saved {} PCA components from {} masks to {}".format(n_components, x.shape[0], opt.out))


if __name__ == "__main__":
    main()
