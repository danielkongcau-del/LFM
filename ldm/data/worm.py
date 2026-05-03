import os
import random

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _as_extensions(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",")]
    extensions = []
    for ext in value:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        extensions.append(ext)
    return tuple(extensions)


class WormPairsBase(Dataset):
    def __init__(
            self,
            root="data/worm",
            split="train",
            size=None,
            crop_size=None,
            random_crop=False,
            flip_p=0.0,
            interpolation="bicubic",
            image_dir_name="image",
            mask_dir_name="mask",
            image_extensions=None,
            mask_extensions=None,
    ):
        self.root = root
        self.split = split
        self.size = size if size is not None else (crop_size if crop_size is not None else 256)
        self.random_crop = random_crop
        self.flip_p = flip_p
        self.image_dir = self._find_dir(root, split, image_dir_name, aliases=("images",))
        self.mask_dir = self._find_dir(root, split, mask_dir_name, aliases=("masks",))
        self.interpolation = {
            "nearest": PIL.Image.NEAREST,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        image_extensions = _as_extensions(image_extensions, IMAGE_EXTENSIONS)
        mask_extensions = _as_extensions(mask_extensions, MASK_EXTENSIONS)
        image_by_stem = self._files_by_stem(self.image_dir, image_extensions)
        mask_by_stem = self._files_by_stem(self.mask_dir, mask_extensions)
        stems = sorted(set(image_by_stem) & set(mask_by_stem), key=self._sort_key)
        if len(stems) == 0:
            raise ValueError("No paired image/mask files found in {}".format(os.path.join(root, split)))

        self.pairs = [(stem, image_by_stem[stem], mask_by_stem[stem]) for stem in stems]
        self.names = [image_name for _, image_name, _ in self.pairs]

        missing_masks = sorted(set(image_by_stem) - set(mask_by_stem), key=self._sort_key)
        missing_images = sorted(set(mask_by_stem) - set(image_by_stem), key=self._sort_key)
        if missing_masks or missing_images:
            print("WormPairsBase: ignoring {} images without masks and {} masks without images under {}.".format(
                len(missing_masks), len(missing_images), os.path.join(root, split)
            ))

    @staticmethod
    def _find_dir(root, split, preferred, aliases=()):
        candidates = [preferred] + [alias for alias in aliases if alias != preferred]
        for name in candidates:
            path = os.path.join(root, split, name)
            if os.path.isdir(path):
                return path
        raise ValueError("Missing directory under {}: expected one of {}".format(
            os.path.join(root, split), candidates
        ))

    @staticmethod
    def _files_by_stem(directory, extensions):
        out = {}
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in extensions:
                continue
            if stem in out:
                raise ValueError("Duplicate file stem '{}' under {}".format(stem, directory))
            out[stem] = name
        return out

    @staticmethod
    def _sort_key(name):
        stem = os.path.splitext(name)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def __len__(self):
        return len(self.names)

    def _crop_pair(self, image, mask):
        width, height = image.size
        crop = min(width, height)
        if self.random_crop and width != height:
            left = random.randint(0, width - crop)
            top = random.randint(0, height - crop)
        else:
            left = (width - crop) // 2
            top = (height - crop) // 2
        box = (left, top, left + crop, top + crop)
        return image.crop(box), mask.crop(box)

    def __getitem__(self, idx):
        stem, image_name, mask_name = self.pairs[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            raise ValueError("Mismatched worm pair sizes for {}: image={}, mask={}".format(
                stem, image.size, mask.size
            ))

        image, mask = self._crop_pair(image, mask)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            mask = mask.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        if self.flip_p > 0.0 and random.random() < self.flip_p:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        mask = (np.array(mask).astype(np.float32) > 127.5).astype(np.float32)
        mask = mask[..., None] * 2.0 - 1.0

        return {
            "image": image,
            "mask": mask,
            "relative_file_path_": image_name,
            "file_path_": image_path,
            "mask_path_": mask_path,
            "pair_stem_": stem,
        }


class WormPairsTrain(WormPairsBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", True)
        kwargs.setdefault("flip_p", 0.5)
        super().__init__(split="train", **kwargs)


class WormPairsValidation(WormPairsBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", False)
        kwargs.setdefault("flip_p", 0.0)
        super().__init__(split="val", **kwargs)


class WormPairsTest(WormPairsBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", False)
        kwargs.setdefault("flip_p", 0.0)
        super().__init__(split="test", **kwargs)


class WormJointConcatBase(WormPairsBase):
    """Worm image/mask pairs with an additional 4-channel image+mask tensor."""

    def __init__(self, concat_key="image_mask", **kwargs):
        super().__init__(**kwargs)
        self.concat_key = concat_key

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out[self.concat_key] = np.concatenate([out["image"], out["mask"]], axis=-1)
        return out


class WormJointConcatTrain(WormJointConcatBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", True)
        kwargs.setdefault("flip_p", 0.5)
        super().__init__(split="train", **kwargs)


class WormJointConcatValidation(WormJointConcatBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", False)
        kwargs.setdefault("flip_p", 0.0)
        super().__init__(split="val", **kwargs)


class WormJointConcatTest(WormJointConcatBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("random_crop", False)
        kwargs.setdefault("flip_p", 0.0)
        super().__init__(split="test", **kwargs)


class WormMaskTokenCache(Dataset):
    """Dataset for cached VQ mask tokens saved as one .npz file per sample."""

    def __init__(
            self,
            root,
            token_key="idx",
            flip_key="idx_flip",
            use_flip=True,
    ):
        self.root = os.path.expanduser(root)
        self.token_key = token_key
        self.flip_key = flip_key
        self.use_flip = use_flip

        if not os.path.isdir(self.root):
            raise ValueError("Missing token cache directory: {}".format(self.root))

        self.files = sorted(name for name in os.listdir(self.root) if name.endswith(".npz"))
        if len(self.files) == 0:
            raise ValueError("No .npz token cache files found in {}".format(self.root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        path = os.path.join(self.root, filename)
        data = np.load(path, allow_pickle=False)

        key = self.token_key
        if self.use_flip and self.flip_key in data.files and random.random() < 0.5:
            key = self.flip_key

        if key not in data.files:
            raise KeyError("Missing key '{}' in token cache file {}".format(key, path))

        idx = np.asarray(data[key], dtype=np.int64).reshape(-1)
        out = {
            "idx": torch.from_numpy(idx),
            "cache_path": path,
        }
        if "grid_size" in data.files:
            out["grid_size"] = torch.as_tensor(int(np.asarray(data["grid_size"]).item()), dtype=torch.long)
        return out
