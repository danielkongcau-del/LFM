import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--threshold", type=float, default=127.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()

    payload = np.load(args.npz)
    samples = payload["arr_0"]
    if args.limit is not None:
        samples = samples[:args.limit]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        if sample.ndim != 3:
            raise ValueError("Expected HWC sample, got shape {}".format(sample.shape))
        raw = np.clip(sample, 0, 255).astype(np.uint8)
        gray = raw.mean(axis=2)
        mask = (gray > args.threshold).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(outdir / "{:06d}_mask.png".format(index))
        if args.save_raw:
            Image.fromarray(raw, mode="RGB").save(raw_dir / "{:06d}_raw.png".format(index))

    print("Wrote {} thresholded masks to {}".format(len(samples), outdir))


if __name__ == "__main__":
    main()
