import os
import random

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset


class WormPairsBase(Dataset):
    def __init__(
            self,
            root="data/worm",
            split="train",
            size=256,
            random_crop=False,
            flip_p=0.0,
            interpolation="bicubic",
    ):
        self.root = root
        self.split = split
        self.size = size
        self.random_crop = random_crop
        self.flip_p = flip_p
        self.image_dir = os.path.join(root, split, "image")
        self.mask_dir = os.path.join(root, split, "mask")
        self.interpolation = {
            "nearest": PIL.Image.NEAREST,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        if not os.path.isdir(self.image_dir):
            raise ValueError("Missing worm image directory: {}".format(self.image_dir))
        if not os.path.isdir(self.mask_dir):
            raise ValueError("Missing worm mask directory: {}".format(self.mask_dir))

        image_names = {n for n in os.listdir(self.image_dir) if n.lower().endswith(".png")}
        mask_names = {n for n in os.listdir(self.mask_dir) if n.lower().endswith(".png")}
        self.names = sorted(image_names & mask_names, key=self._sort_key)
        if len(self.names) == 0:
            raise ValueError("No paired worm image/mask files found in {}".format(os.path.join(root, split)))

        missing_masks = sorted(image_names - mask_names)
        missing_images = sorted(mask_names - image_names)
        if missing_masks or missing_images:
            print("WormPairsBase: ignoring {} images without masks and {} masks without images.".format(
                len(missing_masks), len(missing_images)
            ))

    @staticmethod
    def _sort_key(name):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else stem

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
        name = self.names[idx]
        image_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            raise ValueError("Mismatched worm pair sizes for {}: image={}, mask={}".format(
                name, image.size, mask.size
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
            "relative_file_path_": name,
            "file_path_": image_path,
            "mask_path_": mask_path,
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
