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


def save_pair(image_tensor, mask_tensor, path):
    image = tensor_to_pil(image_tensor)
    mask = tensor_to_pil(mask_tensor)
    canvas = Image.new("RGB", (image.width + mask.width, max(image.height, mask.height)), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(mask, (image.width, 0))
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_config", default="configs/latent-diffusion/worm-mask-ldm-kl-f4.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--image_config", default="configs/latent-diffusion/worm-image-control-mask-ldm-vqkl-f4.yaml")
    parser.add_argument("--image_ckpt", required=True)
    parser.add_argument("--outdir", default="outputs/worm-two-stage")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
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
    mask_model = load_model(opt.mask_config, opt.mask_ckpt, device)
    image_model = load_model(opt.image_config, opt.image_ckpt, device)

    written = 0
    while written < opt.n_samples:
        batch_size = min(opt.batch_size, opt.n_samples - written)

        with mask_model.ema_scope("Two-stage mask sampling"):
            mask_z, _ = mask_model.sample_log(
                cond=None,
                batch_size=batch_size,
                ddim=True,
                ddim_steps=opt.ddim_steps,
                eta=opt.ddim_eta,
            )

        with image_model.ema_scope("Two-stage image sampling"):
            image_z, _ = image_model.sample_log(
                cond=mask_z,
                batch_size=batch_size,
                ddim=True,
                ddim_steps=opt.ddim_steps,
                eta=opt.ddim_eta,
            )

        decoded_mask = mask_model.decode_first_stage(mask_z)
        masks = mask_model.first_stage_model.mask_to_rgb(
            mask_model.first_stage_model.logits_to_mask(decoded_mask["mask_logits"])
        )
        images = image_model.decode_first_stage(image_z)

        for i in range(batch_size):
            index = written + i
            tensor_to_pil(images[i]).save(os.path.join(opt.outdir, "{:06d}_image.png".format(index)))
            tensor_to_pil(masks[i]).save(os.path.join(opt.outdir, "{:06d}_mask.png".format(index)))
            save_pair(images[i], masks[i], os.path.join(opt.outdir, "{:06d}_pair.png".format(index)))

        written += batch_size
        print("Wrote {}/{} samples to {}".format(written, opt.n_samples, opt.outdir))


if __name__ == "__main__":
    main()
