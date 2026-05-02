import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


DEFAULT_IGNORE_KEYS = [
    "model.diffusion_model.label_emb",
    "model_ema.diffusion_modellabel_emb",
]


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp((tensor + 1.0) * 0.5, 0.0, 1.0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def mask_to_pil(mask_channel):
    mask = torch.where(mask_channel > 0.0, torch.ones_like(mask_channel), -torch.ones_like(mask_channel))
    return tensor_to_pil(mask)


def make_grid(rows, labels, out_path, padding=8, label_width=120):
    if not rows:
        return
    cell_w, cell_h = rows[0][0].size
    width = label_width + len(rows[0]) * (cell_w + padding) + padding
    height = len(rows) * (cell_h + padding) + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = padding
    for label, row in zip(labels, rows):
        draw.text((padding, y + 6), label, fill=(0, 0, 0))
        x = label_width
        for image in row:
            sheet.paste(image, (x, y))
            x += cell_w + padding
        y += cell_h + padding
    sheet.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/latent-diffusion/worm-legacy-joint-ldm-uncond-f4.yaml"],
        help="One or more configs, merged from left to right.",
    )
    parser.add_argument("--ckpt", help="Checkpoint to load after instantiating the config.")
    parser.add_argument("--outdir", default="outputs/worm-legacy-joint-ldm-samples")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ignore_key", action="append", default=list(DEFAULT_IGNORE_KEYS))
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    configs = [OmegaConf.load(path) for path in args.config]
    config = OmegaConf.merge(*configs)
    if args.ckpt:
        config.model.params.ckpt_path = None

    device = torch.device(args.device)
    model = instantiate_from_config(config.model)
    if args.ckpt:
        model.init_from_ckpt(args.ckpt, ignore_keys=args.ignore_key)
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    written = 0
    grid_images = []
    grid_masks = []
    with model.ema_scope("Sampling"):
        while written < args.n_samples:
            batch_size = min(args.batch_size, args.n_samples - written)
            samples, _ = sampler.sample(
                S=args.ddim_steps,
                batch_size=batch_size,
                shape=(model.channels, model.image_size, model.image_size),
                conditioning=None,
                eta=args.ddim_eta,
                verbose=False,
            )
            decoded = model.decode_first_stage(samples)
            images, masks = model.split_joint(decoded)

            for i in range(batch_size):
                index = written + i
                image_pil = tensor_to_pil(images[i])
                mask_pil = mask_to_pil(masks[i])
                image_pil.save(os.path.join(args.outdir, "{:06d}_image.png".format(index)))
                mask_pil.save(os.path.join(args.outdir, "{:06d}_mask.png".format(index)))
                grid_images.append(image_pil)
                grid_masks.append(mask_pil)

            written += batch_size
            print("Wrote {}/{} joint samples to {}".format(written, args.n_samples, args.outdir))

    max_grid = min(args.n_samples, 16)
    make_grid(
        [grid_images[:max_grid], grid_masks[:max_grid]],
        ["image", "mask"],
        os.path.join(args.outdir, "grid.png"),
    )


if __name__ == "__main__":
    main()
