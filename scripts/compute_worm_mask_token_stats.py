import argparse
from pathlib import Path

import numpy as np
import torch


def build_token_weights(counts, power, min_weight, max_weight):
    counts = counts.float()
    used = counts > 0
    weights = torch.zeros_like(counts)
    if used.any():
        used_counts = counts[used]
        weights[used] = (used_counts.mean() / used_counts).pow(power)
        weights[used] = weights[used].clamp(min_weight, max_weight)
        weights[used] = weights[used] / weights[used].mean().clamp_min(1.0e-12)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--include_flip", action="store_true")
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--min_weight", type=float, default=0.25)
    parser.add_argument("--max_weight", type=float, default=4.0)
    args = parser.parse_args()

    paths = sorted(Path(args.cache_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError("No .npz token cache files found under {}".format(args.cache_root))

    counts = torch.zeros(args.vocab_size, dtype=torch.long)
    total = 0
    for path in paths:
        payload = np.load(path)
        arrays = [payload["idx"]]
        if args.include_flip and "idx_flip" in payload:
            arrays.append(payload["idx_flip"])
        for array in arrays:
            idx = torch.from_numpy(array.astype(np.int64)).view(-1)
            if idx.numel() == 0:
                continue
            if int(idx.min()) < 0 or int(idx.max()) >= args.vocab_size:
                raise ValueError(
                    "{} contains token ids outside [0, {}].".format(path, args.vocab_size - 1)
                )
            counts += torch.bincount(idx, minlength=args.vocab_size)
            total += idx.numel()

    weights = build_token_weights(
        counts=counts,
        power=args.power,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    freq = counts.float() / max(total, 1)
    used = counts > 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "counts": counts,
            "freq": freq,
            "weights": weights,
            "vocab_size": args.vocab_size,
            "power": args.power,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "include_flip": args.include_flip,
        },
        out,
    )

    topk = min(10, args.vocab_size)
    top_counts, top_ids = torch.topk(counts, topk)
    print("files: {}".format(len(paths)))
    print("total tokens: {}".format(total))
    print("used codes: {}".format(int(used.sum())))
    print("top{}: {}".format(topk, list(zip(top_ids.tolist(), top_counts.tolist()))))
    print("top1 fraction: {:.6f}".format(float(top_counts[0]) / max(total, 1)))
    if used.any():
        used_weights = weights[used]
        print(
            "weight range used: {:.4f} .. {:.4f}".format(
                float(used_weights.min()), float(used_weights.max())
            )
        )
    print("saved: {}".format(out))


if __name__ == "__main__":
    main()
