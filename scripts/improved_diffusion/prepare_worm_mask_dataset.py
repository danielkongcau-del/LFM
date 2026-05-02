import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_ALIASES = {
    "validation": "val",
}


def resolve_split(root, split):
    src_dir = Path(root) / split / "mask"
    if src_dir.is_dir():
        return split, src_dir

    alias = SPLIT_ALIASES.get(split)
    if alias is not None:
        alias_src_dir = Path(root) / alias / "mask"
        if alias_src_dir.is_dir():
            return alias, alias_src_dir

    raise FileNotFoundError("Missing mask directory: {}".format(src_dir))


def prepare_split(root, out_root, split, size):
    source_split, src_dir = resolve_split(root, split)

    dst_dir = Path(out_root) / source_split
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not paths:
        raise FileNotFoundError("No mask images found in {}".format(src_dir))

    for index, path in enumerate(paths):
        mask = Image.open(path).convert("L")
        if size is not None and mask.size != (size, size):
            mask = mask.resize((size, size), resample=Image.NEAREST)
        arr = (np.array(mask, dtype=np.float32) > 127.5).astype(np.uint8) * 255
        out = Image.fromarray(arr, mode="L").convert("RGB")
        out.save(dst_dir / "{:06d}.png".format(index))

    print("Wrote {} masks to {}".format(len(paths), dst_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/worm")
    parser.add_argument("--out_root", default="data/worm-improved-diffusion-mask256")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    for split in args.splits:
        prepare_split(args.root, args.out_root, split, args.size)


if __name__ == "__main__":
    main()
