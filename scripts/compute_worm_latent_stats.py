import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def add_repo_paths():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    taming_path = os.path.join(repo_root, "taming-transformers")
    if os.path.isdir(os.path.join(taming_path, "taming")) and taming_path not in sys.path:
        sys.path.append(taming_path)
    return repo_root


REPO_ROOT = add_repo_paths()

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def trusted_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_init_checkpoint(model_cfg):
    params = model_cfg.get("params", None)
    if params is not None and "ckpt_path" in params:
        params.ckpt_path = None


def load_model_from_run(run_dir, device):
    project_config = os.path.join(run_dir, "configs", "project.yaml")
    checkpoint = os.path.join(run_dir, "checkpoints", "last.ckpt")
    if not os.path.exists(project_config):
        raise FileNotFoundError("Missing project config: {}".format(project_config))
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))

    config = OmegaConf.load(project_config)
    strip_init_checkpoint(config.model)
    model = instantiate_from_config(config.model)
    payload = trusted_torch_load(checkpoint, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded {} with {} missing and {} unexpected keys".format(
        checkpoint, len(missing), len(unexpected)
    ))
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, config


def normalize_split(split):
    return "validation" if split == "val" else split


def build_dataset(config, split, data_root=None, size=None):
    data_cfg = config.data
    split = normalize_split(split)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg.params[split], resolve=True))
    if "params" not in dataset_cfg or dataset_cfg.params is None:
        dataset_cfg.params = OmegaConf.create()
    if data_root is not None:
        dataset_cfg.params.root = data_root
    if size is not None:
        dataset_cfg.params.size = size
    return instantiate_from_config(dataset_cfg)


def batch_to_tensor(batch, key, device):
    tensor = batch[key]
    if len(tensor.shape) == 3:
        tensor = tensor[..., None]
    return tensor.permute(0, 3, 1, 2).to(device=device, memory_format=torch.contiguous_format).float()


@torch.no_grad()
def encode_image_latent(model, image):
    encoded = model.encode(image)
    if isinstance(encoded, torch.Tensor):
        return encoded
    if isinstance(encoded, DiagonalGaussianDistribution):
        return encoded.mode()
    if isinstance(encoded, (tuple, list)) and len(encoded) > 0 and isinstance(encoded[0], torch.Tensor):
        return encoded[0]
    raise TypeError("Unsupported image encoder output type: {}".format(type(encoded)))


@torch.no_grad()
def encode_mask_latent(model, mask, sample=False):
    encoded = model.encode(mask)
    if isinstance(encoded, DiagonalGaussianDistribution):
        return encoded.sample() if sample else encoded.mode()
    if isinstance(encoded, torch.Tensor):
        return encoded
    if isinstance(encoded, (tuple, list)) and len(encoded) > 0 and isinstance(encoded[0], torch.Tensor):
        return encoded[0]
    raise TypeError("Unsupported mask encoder output type: {}".format(type(encoded)))


class RunningTensorStats:
    def __init__(self, channels):
        self.channels = channels
        self.count = 0
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sq_sum = torch.zeros(channels, dtype=torch.float64)
        self.global_sum = 0.0
        self.global_sq_sum = 0.0
        self.global_count = 0
        self.min = None
        self.max = None
        self.shape = None

    def update(self, tensor):
        tensor = tensor.detach().cpu().double()
        if tensor.ndim != 4:
            raise ValueError("Expected BCHW tensor, got shape {}".format(tuple(tensor.shape)))
        if tensor.shape[1] != self.channels:
            raise ValueError("Expected {} channels, got {}".format(self.channels, tensor.shape[1]))
        self.shape = list(tensor.shape[1:])
        reduce_dims = (0, 2, 3)
        self.sum += tensor.sum(dim=reduce_dims)
        self.sq_sum += (tensor ** 2).sum(dim=reduce_dims)
        values_per_channel = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
        self.count += values_per_channel
        self.global_sum += tensor.sum().item()
        self.global_sq_sum += (tensor ** 2).sum().item()
        self.global_count += tensor.numel()
        current_min = tensor.amin(dim=reduce_dims)
        current_max = tensor.amax(dim=reduce_dims)
        self.min = current_min if self.min is None else torch.minimum(self.min, current_min)
        self.max = current_max if self.max is None else torch.maximum(self.max, current_max)

    @staticmethod
    def _std_from_sums(sum_value, sq_sum_value, count):
        mean = sum_value / count
        variance = sq_sum_value / count - mean ** 2
        variance = torch.clamp(variance, min=0.0)
        return torch.sqrt(variance)

    def as_dict(self):
        channel_mean = self.sum / self.count
        channel_std = self._std_from_sums(self.sum, self.sq_sum, self.count)
        global_mean = self.global_sum / self.global_count
        global_variance = self.global_sq_sum / self.global_count - global_mean ** 2
        global_std = max(global_variance, 0.0) ** 0.5
        return OrderedDict([
            ("shape_chw", self.shape),
            ("global_mean", global_mean),
            ("global_std", global_std),
            ("channel_mean", channel_mean.tolist()),
            ("channel_std", channel_std.tolist()),
            ("channel_min", self.min.tolist()),
            ("channel_max", self.max.tolist()),
            ("num_values_per_channel", self.count),
        ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-run", "--image-ae-dir", dest="image_run",
                        default="logs/autoencoder/worm-image-vq-f4-noattn-ft-b4")
    parser.add_argument("--mask-run", "--mask-ae-dir", dest="mask_run",
                        default="logs/autoencoder/worm-mask-kl_f4-kl1e7")
    parser.add_argument("--split", default="train", choices=["train", "validation", "val", "test"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 means all batches.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask-sample", type=str2bool, default=False)
    parser.add_argument("--output", default="logs/autoencoder/worm-latent-stats/latent_stats.json")
    parser.add_argument("--save-preview", type=str2bool, default=True)
    parser.add_argument("--preview-count", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_run = os.path.normpath(args.image_run)
    mask_run = os.path.normpath(args.mask_run)
    output_path = os.path.normpath(args.output)

    image_model, image_config = load_model_from_run(image_run, device)
    mask_model, _ = load_model_from_run(mask_run, device)
    dataset = build_dataset(image_config, args.split, data_root=args.data_root, size=args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    image_stats = None
    mask_stats = None
    joint_stats = None
    preview = None
    num_samples = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        image = batch_to_tensor(batch, "image", device)
        mask = batch_to_tensor(batch, "mask", device)
        image_z = encode_image_latent(image_model, image)
        mask_z = encode_mask_latent(mask_model, mask, sample=args.mask_sample)
        if image_z.shape[2:] != mask_z.shape[2:]:
            raise ValueError("Image and mask latent spatial sizes differ: {} vs {}".format(
                tuple(image_z.shape), tuple(mask_z.shape)
            ))

        joint_z = torch.cat([image_z, mask_z], dim=1)
        if image_stats is None:
            image_stats = RunningTensorStats(image_z.shape[1])
            mask_stats = RunningTensorStats(mask_z.shape[1])
            joint_stats = RunningTensorStats(joint_z.shape[1])
        image_stats.update(image_z)
        mask_stats.update(mask_z)
        joint_stats.update(joint_z)
        num_samples += image_z.shape[0]

        if args.save_preview and preview is None:
            take = min(args.preview_count, image_z.shape[0])
            preview = {
                "image_z": image_z[:take].detach().cpu(),
                "mask_z": mask_z[:take].detach().cpu(),
                "joint_z": joint_z[:take].detach().cpu(),
                "file_path": batch.get("file_path_", [])[:take],
                "mask_path": batch.get("mask_path_", [])[:take],
            }

        print("Processed batch {} | samples {}".format(batch_idx + 1, num_samples))

    if image_stats is None:
        raise RuntimeError("No batches processed.")

    result = OrderedDict([
        ("image_run", image_run),
        ("mask_run", mask_run),
        ("split", normalize_split(args.split)),
        ("data_root_override", args.data_root),
        ("size_override", args.size),
        ("batch_size", args.batch_size),
        ("max_batches", args.max_batches),
        ("num_samples", num_samples),
        ("mask_sample", args.mask_sample),
        ("image_latent", image_stats.as_dict()),
        ("mask_latent", mask_stats.as_dict()),
        ("joint_latent", joint_stats.as_dict()),
    ])
    result["recommended_normalization"] = OrderedDict([
        ("image", OrderedDict([
            ("mean", result["image_latent"]["channel_mean"]),
            ("std", result["image_latent"]["channel_std"]),
        ])),
        ("mask", OrderedDict([
            ("mean", result["mask_latent"]["channel_mean"]),
            ("std", result["mask_latent"]["channel_std"]),
        ])),
        ("joint", OrderedDict([
            ("mean", result["joint_latent"]["channel_mean"]),
            ("std", result["joint_latent"]["channel_std"]),
        ])),
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Saved stats to {}".format(output_path))

    if preview is not None:
        preview_path = os.path.splitext(output_path)[0] + "_preview.pt"
        torch.save(preview, preview_path)
        print("Saved preview latents to {}".format(preview_path))


if __name__ == "__main__":
    main()
