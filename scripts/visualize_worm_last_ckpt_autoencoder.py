import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldm.data.worm import WormJointConcatBase
from ldm.util import instantiate_from_config


def tensor_to_pil(tensor, rescale=True):
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if rescale:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def make_sheet(rows, out_path):
    font = ImageFont.load_default()
    padding = 8
    label_width = 170
    rendered_rows = []
    sheet_width = 0

    for label, images in rows:
        if not images:
            continue
        height = max(image.height for image in images)
        width = sum(image.width for image in images) + padding * (len(images) - 1)
        row = Image.new("RGB", (label_width + width + padding * 3, height + padding * 2), "white")
        draw = ImageDraw.Draw(row)
        draw.rectangle((0, 0, row.width - 1, row.height - 1), outline=(210, 210, 210))
        draw.text((padding, padding), label, fill=(0, 0, 0), font=font)
        x = label_width + padding * 2
        for image in images:
            y = padding + (height - image.height) // 2
            row.paste(image, (x, y))
            draw.rectangle((x, y, x + image.width - 1, y + image.height - 1), outline=(160, 160, 160))
            x += image.width + padding
        rendered_rows.append(row)
        sheet_width = max(sheet_width, row.width)

    if not rendered_rows:
        raise ValueError("No images to render.")

    title_height = 28
    sheet_height = title_height + padding + sum(row.height for row in rendered_rows) + padding * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 7), "legacy 4ch AutoencoderKL reconstruction", fill=(0, 0, 0), font=font)
    y = title_height + padding
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + padding
    sheet.save(out_path)


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    if ckpt_path is not None:
        config.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(config.model)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/autoencoder/worm-last-ckpt-joint-kl-f4.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--root", default="data/worm")
    parser.add_argument("--split", default="val")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--concat_key", default="image_mask")
    parser.add_argument("--outdir", default="outputs/worm-last-ckpt-autoencoder")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.ckpt, device)
    dataset = WormJointConcatBase(
        root=args.root,
        split=args.split,
        size=args.size,
        random_crop=False,
        flip_p=0.0,
        concat_key=args.concat_key,
    )
    loader = DataLoader(dataset, batch_size=args.n_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x = model.get_input(batch, model.image_key).to(device)
    reconstruction, _ = model(x, sample_posterior=False)

    input_image, input_mask = model.split_joint(x)
    rec_image, rec_mask = model.split_joint(reconstruction)
    input_mask = model.mask_from_channel(input_mask, binarized=True)
    rec_mask = model.mask_from_channel(rec_mask, binarized=True)

    rows = [
        ("Original image", [tensor_to_pil(tensor) for tensor in input_image]),
        ("Reconstructed image", [tensor_to_pil(tensor) for tensor in rec_image]),
        ("Original mask", [tensor_to_pil(tensor) for tensor in input_mask]),
        ("Reconstructed mask", [tensor_to_pil(tensor) for tensor in rec_mask]),
    ]
    make_sheet(rows, outdir / "comparison.png")

    for index in range(input_image.shape[0]):
        sample_dir = outdir / "{:04d}".format(index)
        sample_dir.mkdir(exist_ok=True)
        tensor_to_pil(input_image[index]).save(sample_dir / "input_image.png")
        tensor_to_pil(rec_image[index]).save(sample_dir / "reconstruction_image.png")
        tensor_to_pil(input_mask[index]).save(sample_dir / "input_mask.png")
        tensor_to_pil(rec_mask[index]).save(sample_dir / "reconstruction_mask.png")

    print("Wrote reconstruction sheet to {}".format(outdir / "comparison.png"))


if __name__ == "__main__":
    main()
