import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_dir(split_dir, names):
    for name in names:
        path = split_dir / name
        if path.is_dir():
            return path
    return None


def collect_pairs(root, source_splits):
    pairs = {}
    duplicates = 0
    for split in source_splits:
        split_dir = root / split
        image_dir = find_dir(split_dir, ("image", "images"))
        mask_dir = find_dir(split_dir, ("mask", "masks"))
        if image_dir is None or mask_dir is None:
            continue

        image_by_stem = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        mask_by_stem = {
            path.stem: path
            for path in mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
        }
        for stem in sorted(set(image_by_stem) & set(mask_by_stem)):
            if stem in pairs:
                duplicates += 1
                continue
            pairs[stem] = (image_by_stem[stem], mask_by_stem[stem])
    return pairs, duplicates


def split_stems(stems, train_ratio, val_ratio, seed):
    stems = list(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    total = len(stems)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def ensure_child(path, parent):
    resolved = path.resolve()
    parent = parent.resolve()
    if resolved.parent != parent:
        raise ValueError("Refusing to operate outside dataset root: {}".format(resolved))


def write_split(root, pairs, split_map, overwrite):
    tmp = root / ".resplit_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    for split, stems in split_map.items():
        image_dir = tmp / split / "image"
        mask_dir = tmp / split / "mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            image_path, mask_path = pairs[stem]
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(mask_path, mask_dir / mask_path.name)

    if overwrite:
        for split in ("train", "val", "test"):
            target = root / split
            ensure_child(target, root)
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(tmp / split), str(target))
        shutil.rmtree(tmp)
    else:
        print("wrote preview split to {}".format(tmp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source_splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1.0e-6:
        raise ValueError("Ratios must sum to 1.0, got {}".format(ratio_sum))

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    pairs, duplicates = collect_pairs(root, args.source_splits)
    if not pairs:
        raise ValueError("No paired image/mask files found under {}".format(root))

    split_map = split_stems(sorted(pairs), args.train_ratio, args.val_ratio, args.seed)
    print("dataset: {}".format(root))
    print("unique pairs: {}".format(len(pairs)))
    print("duplicate stems ignored: {}".format(duplicates))
    for split in ("train", "val", "test"):
        print("{}: {}".format(split, len(split_map[split])))

    write_split(root, pairs, split_map, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
