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


def split_joint(x):
    if x.shape[1] < 4:
        raise ValueError("Expected at least 4 channels, got {}".format(tuple(x.shape)))
    return x[:, :3], x[:, 3:4]


def make_grid(items, out_path, padding=8, label_height=18):
    if not items:
        return
    cell_w, cell_h = items[0][1].size
    width = len(items) * (cell_w + padding) + padding
    height = 2 * (cell_h + label_height + padding) + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = padding
    for label, image, mask in items:
        draw.text((x, padding), label, fill=(0, 0, 0))
        sheet.paste(image, (x, padding + label_height))
        y_mask = padding + label_height + cell_h + padding
        draw.text((x, y_mask), label, fill=(0, 0, 0))
        sheet.paste(mask, (x, y_mask + label_height))
        x += cell_w + padding
    sheet.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent-diffusion/worm-legacy-joint-ldm-original-adm-f4.yaml")
    parser.add_argument("--ckpt", default="logs/generator/original/last.ckpt")
    parser.add_argument("--ae_ckpt", default="logs/autoencoder/joint-AE-original/last.ckpt")
    parser.add_argument("--outdir", default="outputs/original-generator-adm-samples")
    parser.add_argument("--class_label", type=int, action="append", default=[0, 1, 2, 3])
    parser.add_argument("--n_per_class", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OmegaConf.load(args.config)
    config.model.params.ckpt_path = None
    config.model.params.first_stage_config.params.ckpt_path = args.ae_ckpt

    device = torch.device(args.device)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt, ignore_keys=[])
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    grid_items = []
    with model.ema_scope("ADM sampling"):
        for class_label in args.class_label:
            remaining = args.n_per_class
            sample_index = 0
            while remaining > 0:
                batch_size = min(args.batch_size, remaining)
                labels = torch.full((batch_size,), int(class_label), dtype=torch.long, device=device)
                samples, _ = sampler.sample(
                    S=args.ddim_steps,
                    batch_size=batch_size,
                    shape=(model.channels, model.image_size, model.image_size),
                    conditioning=labels,
                    eta=args.ddim_eta,
                    verbose=False,
                )
                decoded = model.decode_first_stage(samples)
                images, masks = split_joint(decoded)

                for i in range(batch_size):
                    label = "class{}_{}".format(class_label, sample_index)
                    image_pil = tensor_to_pil(images[i])
                    mask_pil = mask_to_pil(masks[i])
                    image_pil.save(os.path.join(args.outdir, "{}_image.png".format(label)))
                    mask_pil.save(os.path.join(args.outdir, "{}_mask.png".format(label)))
                    if len(grid_items) < 16:
                        grid_items.append((label, image_pil, mask_pil))
                    sample_index += 1
                remaining -= batch_size
                print("Wrote class {} sample(s), remaining {}".format(class_label, remaining))

    make_grid(grid_items, os.path.join(args.outdir, "grid.png"))


if __name__ == "__main__":
    main()
