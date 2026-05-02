import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    return model


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_config", default="configs/maskgit/worm-maskgit-f8.yaml")
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--vq_config", default="configs/autoencoder/worm-mask-vq.yaml")
    parser.add_argument("--vq_ckpt", required=True)
    parser.add_argument("--outdir", default="outputs/worm-maskgit-samples")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=12)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--gumbel_temp", type=float, default=4.5)
    parser.add_argument("--token_weight_path")
    parser.add_argument("--sample_token_logits_alpha", type=float)
    parser.add_argument("--sample_token_confidence_alpha", type=float)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main():
    opt = parse_args()
    if opt.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.makedirs(opt.outdir, exist_ok=True)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    device = torch.device(opt.device)
    prior = load_model(opt.prior_config, opt.prior_ckpt, device)
    if opt.token_weight_path is not None:
        prior.token_weight_path = opt.token_weight_path
        prior.__dict__["_token_weights"] = None
    if opt.sample_token_logits_alpha is not None:
        prior.sample_token_logits_alpha = opt.sample_token_logits_alpha
    if opt.sample_token_confidence_alpha is not None:
        prior.sample_token_confidence_alpha = opt.sample_token_confidence_alpha
    vq = load_model(opt.vq_config, opt.vq_ckpt, device)
    vq.requires_grad_(False)

    written = 0
    while written < opt.n_samples:
        batch_size = min(opt.batch_size, opt.n_samples - written)
        indices = prior.sample_indices(
            batch_size,
            sampling_steps=opt.sampling_steps,
            topk=opt.topk,
            softmax_temp=opt.softmax_temp,
            gumbel_temp=opt.gumbel_temp,
        )
        indices = indices.view(batch_size, prior.grid_size, prior.grid_size)
        logits = vq.decode_code(indices)
        probs = torch.sigmoid(logits)
        masks = torch.where(probs > 0.5, torch.ones_like(probs), -torch.ones_like(probs))

        for i in range(batch_size):
            index = written + i
            tensor_to_pil(masks[i]).save(os.path.join(opt.outdir, "{:06d}_mask.png".format(index)))

        written += batch_size
        print("Wrote {}/{} masks to {}".format(written, opt.n_samples, opt.outdir))


if __name__ == "__main__":
    main()
