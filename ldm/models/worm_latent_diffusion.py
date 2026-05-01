import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


def trusted_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_init_checkpoint(model_cfg):
    params = model_cfg.get("params", None)
    if params is not None and "ckpt_path" in params:
        params.ckpt_path = None


def tensor_from_channels(values, channels):
    if values is None:
        values = [0.0] * channels
    if len(values) != channels:
        raise ValueError("Expected {} channel values, got {}".format(channels, len(values)))
    return torch.tensor(values, dtype=torch.float32).view(1, channels, 1, 1)


def load_model_from_run(run_dir):
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
    model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    return model


class WormJointFirstStage(nn.Module):
    def __init__(
            self,
            image_run="logs/autoencoder/worm-image-vq-f4-noattn-ft-b4",
            mask_run="logs/autoencoder/worm-mask-kl_f4-kl1e7",
            image_key="image",
            mask_key="mask",
            image_channels=3,
            mask_channels=4,
            channel_mean=None,
            channel_std=None,
            normalize=True,
            mask_sample=False,
    ):
        super().__init__()
        self.image_key = image_key
        self.mask_key = mask_key
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.channels = image_channels + mask_channels
        self.normalize = normalize
        self.mask_sample = mask_sample

        self.image_model = load_model_from_run(image_run)
        self.mask_model = load_model_from_run(mask_run)

        self.register_buffer("channel_mean", tensor_from_channels(channel_mean, self.channels))
        self.register_buffer("channel_std", tensor_from_channels(channel_std, self.channels))

    @staticmethod
    def batch_to_tensor(batch, key, device=None):
        tensor = batch[key]
        if len(tensor.shape) == 3:
            tensor = tensor[..., None]
        tensor = tensor.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def batch_to_inputs(self, batch, device=None):
        image = self.batch_to_tensor(batch, self.image_key, device=device)
        mask = self.batch_to_tensor(batch, self.mask_key, device=device)
        return image, mask

    @staticmethod
    def latent_from_encoder_output(encoded, sample=False):
        if isinstance(encoded, DiagonalGaussianDistribution):
            return encoded.sample() if sample else encoded.mode()
        if isinstance(encoded, torch.Tensor):
            return encoded
        if isinstance(encoded, (tuple, list)) and len(encoded) > 0 and isinstance(encoded[0], torch.Tensor):
            return encoded[0]
        raise TypeError("Unsupported encoder output type: {}".format(type(encoded)))

    def normalize_latent(self, z):
        if not self.normalize:
            return z
        return (z - self.channel_mean.to(z)) / torch.clamp(self.channel_std.to(z), min=1.0e-6)

    def denormalize_latent(self, z):
        if not self.normalize:
            return z
        return z * torch.clamp(self.channel_std.to(z), min=1.0e-6) + self.channel_mean.to(z)

    @torch.no_grad()
    def encode(self, image, mask):
        image_z = self.latent_from_encoder_output(self.image_model.encode(image), sample=False)
        mask_z = self.latent_from_encoder_output(self.mask_model.encode(mask), sample=self.mask_sample)
        if image_z.shape[2:] != mask_z.shape[2:]:
            raise ValueError("Image and mask latent spatial sizes differ: {} vs {}".format(
                tuple(image_z.shape), tuple(mask_z.shape)
            ))
        z = torch.cat([image_z, mask_z], dim=1)
        return self.normalize_latent(z)

    @torch.no_grad()
    def decode(self, z):
        z = self.denormalize_latent(z)
        image_z = z[:, :self.image_channels]
        mask_z = z[:, self.image_channels:self.image_channels + self.mask_channels]
        return {
            "image": self.image_model.decode(image_z),
            "mask_logits": self.mask_model.decode(mask_z),
        }

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    @staticmethod
    def logits_to_mask(logits):
        return torch.sigmoid(logits) * 2.0 - 1.0


class WormJointLatentDiffusion(LatentDiffusion):
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        if bs is not None:
            image = image[:bs]
            mask = mask[:bs]
        z = self.first_stage_model.encode(image, mask).detach()

        if self.model.conditioning_key is not None:
            raise NotImplementedError("WormJointLatentDiffusion currently expects unconditional training.")

        out = [z, None]
        if return_first_stage_outputs:
            decoded = self.decode_first_stage(z)
            out.extend([image, decoded["image"]])
        if return_original_cond:
            out.append(None)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, *args, **kwargs):
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=50, ddim_eta=1.0,
                   plot_diffusion_rows=False, plot_progressive_rows=False, **kwargs):
        log = dict()
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        N = min(image.shape[0], N)
        image = image[:N]
        mask = mask[:N]
        z = self.first_stage_model.encode(image, mask)
        decoded = self.decode_first_stage(z)

        log["inputs_image"] = image
        log["inputs_mask"] = self.first_stage_model.mask_to_rgb(mask)
        log["reconstructions_image"] = decoded["image"]
        log["reconstructions_mask"] = self.first_stage_model.mask_to_rgb(
            self.first_stage_model.logits_to_mask(decoded["mask_logits"])
        )

        if sample:
            use_ddim = ddim_steps is not None
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(
                    cond=None,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            decoded_samples = self.decode_first_stage(samples)
            log["samples_image"] = decoded_samples["image"]
            log["samples_mask"] = self.first_stage_model.mask_to_rgb(
                self.first_stage_model.logits_to_mask(decoded_samples["mask_logits"])
            )
        return log
