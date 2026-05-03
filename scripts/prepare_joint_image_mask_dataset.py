import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def rename_dir(split_dir, old_name, new_name):
    old = split_dir / old_name
    new = split_dir / new_name
    if not old.exists():
        return False
    if new.exists():
        raise FileExistsError("Both {} and {} exist; resolve this manually.".format(old, new))
    old.rename(new)
    return True


def copy_test_to_train_if_needed(root):
    train = root / "train"
    test = root / "test"
    if train.exists() or not test.exists():
        return False
    shutil.copytree(test, train)
    return True


def normalize_layout(root):
    renamed = []
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if rename_dir(split_dir, "images", "image"):
            renamed.append(str(split_dir / "image"))
        if rename_dir(split_dir, "masks", "mask"):
            renamed.append(str(split_dir / "mask"))
    return renamed


def normalize_masks(root):
    converted = 0
    already_binary = 0
    other = []
    for mask_dir in sorted(root.glob("*/mask")):
        if not mask_dir.is_dir():
            continue
        for path in sorted(mask_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                image = Image.open(path)
            except OSError:
                continue
            arr = np.array(image)
            if arr.ndim == 3:
                arr = arr[..., 0]
            values = set(np.unique(arr).tolist())
            if values.issubset({0, 1}):
                out = (arr > 0).astype(np.uint8) * 255
                Image.fromarray(out, mode="L").save(path)
                converted += 1
            elif values.issubset({0, 255}):
                already_binary += 1
            else:
                other.append(str(path))
    return converted, already_binary, other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="Dataset roots such as data/worm data/weed data/mango.")
    parser.add_argument(
        "--copy_test_to_train_if_missing",
        action="store_true",
        help="For datasets without train/, copy test/ to train/ without modifying test/.",
    )
    args = parser.parse_args()

    for root_arg in args.roots:
        root = Path(root_arg)
        if not root.exists():
            raise FileNotFoundError(root)

        copied_train = False
        if args.copy_test_to_train_if_missing:
            copied_train = copy_test_to_train_if_needed(root)
        renamed = normalize_layout(root)
        converted, already_binary, other = normalize_masks(root)

        print("dataset: {}".format(root))
        print("  copied test -> train: {}".format(copied_train))
        print("  renamed dirs: {}".format(len(renamed)))
        print("  converted 0/1 masks to 0/255: {}".format(converted))
        print("  already 0/255 masks: {}".format(already_binary))
        if other:
            print("  masks with non-binary values: {}".format(len(other)))
            for path in other[:10]:
                print("    {}".format(path))


if __name__ == "__main__":
    main()
