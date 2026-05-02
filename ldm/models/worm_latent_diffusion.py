import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.modules.ema import LitEma
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

    def normalize_image_latent(self, z):
        if not self.normalize:
            return z
        mean = self.channel_mean[:, :self.image_channels]
        std = self.channel_std[:, :self.image_channels]
        return (z - mean.to(z)) / torch.clamp(std.to(z), min=1.0e-6)

    def denormalize_image_latent(self, z):
        if not self.normalize:
            return z
        mean = self.channel_mean[:, :self.image_channels]
        std = self.channel_std[:, :self.image_channels]
        return z * torch.clamp(std.to(z), min=1.0e-6) + mean.to(z)

    def normalize_mask_latent(self, z):
        if not self.normalize:
            return z
        start = self.image_channels
        end = start + self.mask_channels
        mean = self.channel_mean[:, start:end]
        std = self.channel_std[:, start:end]
        return (z - mean.to(z)) / torch.clamp(std.to(z), min=1.0e-6)

    def denormalize_mask_latent(self, z):
        if not self.normalize:
            return z
        start = self.image_channels
        end = start + self.mask_channels
        mean = self.channel_mean[:, start:end]
        std = self.channel_std[:, start:end]
        return z * torch.clamp(std.to(z), min=1.0e-6) + mean.to(z)

    @torch.no_grad()
    def encode_image(self, image):
        image_z = self.latent_from_encoder_output(self.image_model.encode(image), sample=False)
        return self.normalize_image_latent(image_z)

    @torch.no_grad()
    def encode_mask(self, mask):
        mask_z = self.latent_from_encoder_output(self.mask_model.encode(mask), sample=self.mask_sample)
        return self.normalize_mask_latent(mask_z)

    @torch.no_grad()
    def encode(self, image, mask):
        image_z = self.encode_image(image)
        mask_z = self.encode_mask(mask)
        if image_z.shape[2:] != mask_z.shape[2:]:
            raise ValueError("Image and mask latent spatial sizes differ: {} vs {}".format(
                tuple(image_z.shape), tuple(mask_z.shape)
            ))
        return torch.cat([image_z, mask_z], dim=1)

    @torch.no_grad()
    def decode_image(self, z):
        z = self.denormalize_image_latent(z)
        return self.image_model.decode(z)

    @torch.no_grad()
    def decode_mask_logits(self, z):
        z = self.denormalize_mask_latent(z)
        return self.mask_model.decode(z)

    @torch.no_grad()
    def decode(self, z):
        image_z = z[:, :self.image_channels]
        mask_z = z[:, self.image_channels:self.image_channels + self.mask_channels]
        return {
            "image": self.decode_image(image_z),
            "mask_logits": self.decode_mask_logits(mask_z),
        }

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    @staticmethod
    def logits_to_mask(logits):
        return torch.sigmoid(logits) * 2.0 - 1.0


class WormMaskFirstStage(nn.Module):
    def __init__(
            self,
            mask_run="logs/autoencoder/worm-mask-kl_f4-kl1e7",
            mask_key="mask",
            mask_channels=4,
            channel_mean=None,
            channel_std=None,
            normalize=True,
            mask_sample=False,
    ):
        super().__init__()
        self.mask_key = mask_key
        self.mask_channels = mask_channels
        self.channels = mask_channels
        self.normalize = normalize
        self.mask_sample = mask_sample

        self.mask_model = load_model_from_run(mask_run)

        self.register_buffer("channel_mean", tensor_from_channels(channel_mean, self.channels))
        self.register_buffer("channel_std", tensor_from_channels(channel_std, self.channels))

    @staticmethod
    def batch_to_tensor(batch, key, device=None):
        return WormJointFirstStage.batch_to_tensor(batch, key, device=device)

    def batch_to_mask(self, batch, device=None):
        return self.batch_to_tensor(batch, self.mask_key, device=device)

    @staticmethod
    def latent_from_encoder_output(encoded, sample=False):
        return WormJointFirstStage.latent_from_encoder_output(encoded, sample=sample)

    def normalize_latent(self, z):
        if not self.normalize:
            return z
        return (z - self.channel_mean.to(z)) / torch.clamp(self.channel_std.to(z), min=1.0e-6)

    def denormalize_latent(self, z):
        if not self.normalize:
            return z
        return z * torch.clamp(self.channel_std.to(z), min=1.0e-6) + self.channel_mean.to(z)

    @torch.no_grad()
    def encode(self, mask):
        mask_z = self.latent_from_encoder_output(self.mask_model.encode(mask), sample=self.mask_sample)
        return self.normalize_latent(mask_z)

    @torch.no_grad()
    def decode_mask_logits(self, z):
        z = self.denormalize_latent(z)
        return self.mask_model.decode(z)

    @torch.no_grad()
    def decode(self, z):
        return {"mask_logits": self.decode_mask_logits(z)}

    @staticmethod
    def mask_to_rgb(mask):
        return WormJointFirstStage.mask_to_rgb(mask)

    @staticmethod
    def logits_to_mask(logits):
        return WormJointFirstStage.logits_to_mask(logits)


class WormMaskLatentDiffusion(LatentDiffusion):
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        mask = self.first_stage_model.batch_to_mask(batch, device=self.device)
        if bs is not None:
            mask = mask[:bs]
        z = self.first_stage_model.encode(mask).detach()

        if self.model.conditioning_key is not None:
            raise NotImplementedError("WormMaskLatentDiffusion expects unconditional training.")

        out = [z, None]
        if return_first_stage_outputs:
            decoded = self.decode_first_stage(z)
            out.extend([mask, self.first_stage_model.logits_to_mask(decoded["mask_logits"])])
        if return_original_cond:
            out.append(None)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, *args, **kwargs):
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=100, ddim_eta=0.0,
                   plot_diffusion_rows=False, plot_progressive_rows=False, **kwargs):
        log = dict()
        mask = self.first_stage_model.batch_to_mask(batch, device=self.device)
        N = min(mask.shape[0], N)
        mask = mask[:N]
        z = self.first_stage_model.encode(mask)
        decoded = self.decode_first_stage(z)

        log["inputs_mask"] = self.first_stage_model.mask_to_rgb(mask)
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
            log["samples_mask"] = self.first_stage_model.mask_to_rgb(
                self.first_stage_model.logits_to_mask(decoded_samples["mask_logits"])
            )
        return log


class WormImageMaskConditionedLatentDiffusion(LatentDiffusion):
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        if bs is not None:
            image = image[:bs]
            mask = mask[:bs]

        image_z = self.first_stage_model.encode_image(image).detach()
        mask_z = self.first_stage_model.encode_mask(mask).detach()

        if self.model.conditioning_key != "concat":
            raise NotImplementedError("WormImageMaskConditionedLatentDiffusion expects concat conditioning.")

        out = [image_z, mask_z]
        if return_first_stage_outputs:
            out.extend([image, self.decode_first_stage(image_z)])
        if return_original_cond:
            out.append(mask)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, *args, **kwargs):
        return self.first_stage_model.decode_image(z)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=100, ddim_eta=0.0,
                   plot_diffusion_rows=False, plot_progressive_rows=False, **kwargs):
        log = dict()
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        N = min(image.shape[0], N)
        image = image[:N]
        mask = mask[:N]
        image_z = self.first_stage_model.encode_image(image)
        mask_z = self.first_stage_model.encode_mask(mask)

        log["inputs_image"] = image
        log["inputs_mask"] = self.first_stage_model.mask_to_rgb(mask)
        log["reconstructions_image"] = self.decode_first_stage(image_z)
        log["reconstructions_mask"] = self.first_stage_model.mask_to_rgb(
            self.first_stage_model.logits_to_mask(self.first_stage_model.decode_mask_logits(mask_z))
        )

        if sample:
            use_ddim = ddim_steps is not None
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(
                    cond=mask_z,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            log["samples_image"] = self.decode_first_stage(samples)
        return log


class WormImageMaskControlLatentDiffusion(LatentDiffusion):
    def __init__(self, control_stage_config, only_mid_control=False, control_scales=None, *args, **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(*args, ckpt_path=None, ignore_keys=ignore_keys, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control
        self.control_scales = control_scales

        if self.use_ema:
            self.control_model_ema = LitEma(self.control_model)
            print(f"Keeping EMAs of {len(list(self.control_model_ema.buffers()))} control buffers.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        if bs is not None:
            image = image[:bs]
            mask = mask[:bs]

        image_z = self.first_stage_model.encode_image(image).detach()
        mask_z = self.first_stage_model.encode_mask(mask).detach()

        out = [image_z, mask_z]
        if return_first_stage_outputs:
            out.extend([image, self.decode_first_stage(image_z)])
        if return_original_cond:
            out.append(mask)
        return out

    @staticmethod
    def _hint_from_conditioning(cond):
        if cond is None:
            raise ValueError("WormImageMaskControlLatentDiffusion requires a mask latent condition.")
        if isinstance(cond, dict):
            if "c_concat" not in cond:
                raise KeyError("Expected c_concat in control conditioning dict.")
            cond = cond["c_concat"]
        if isinstance(cond, (tuple, list)):
            return torch.cat(list(cond), dim=1)
        return cond

    def _scaled_control(self, control):
        if self.control_scales is None:
            scales = [1.0] * len(control)
        elif len(self.control_scales) == 1:
            scales = list(self.control_scales) * len(control)
        elif len(self.control_scales) == len(control):
            scales = self.control_scales
        else:
            raise ValueError("Expected 1 or {} control scales, got {}".format(
                len(control), len(self.control_scales)
            ))
        return [c * scale for c, scale in zip(control, scales)]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        hint = self._hint_from_conditioning(cond).to(x_noisy.device)
        diffusion_model = self.model.diffusion_model
        control = self.control_model(
            x=x_noisy,
            hint=hint,
            timesteps=t,
            context=None,
        )
        control = self._scaled_control(control)
        model_output = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=None,
            control=control,
            only_mid_control=self.only_mid_control,
        )
        if isinstance(model_output, tuple) and not return_ids:
            return model_output[0]
        return model_output

    @torch.no_grad()
    def decode_first_stage(self, z, *args, **kwargs):
        return self.first_stage_model.decode_image(z)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            self.control_model_ema.store(self.control_model.parameters())
            self.control_model_ema.copy_to(self.control_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                self.control_model_ema.restore(self.control_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
            self.control_model_ema(self.control_model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(self.control_model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=100, ddim_eta=0.0,
                   plot_diffusion_rows=False, plot_progressive_rows=False, **kwargs):
        log = dict()
        image, mask = self.first_stage_model.batch_to_inputs(batch, device=self.device)
        N = min(image.shape[0], N)
        image = image[:N]
        mask = mask[:N]
        image_z = self.first_stage_model.encode_image(image)
        mask_z = self.first_stage_model.encode_mask(mask)

        log["inputs_image"] = image
        log["inputs_mask"] = self.first_stage_model.mask_to_rgb(mask)
        log["reconstructions_image"] = self.decode_first_stage(image_z)
        log["reconstructions_mask"] = self.first_stage_model.mask_to_rgb(
            self.first_stage_model.logits_to_mask(self.first_stage_model.decode_mask_logits(mask_z))
        )

        if sample:
            use_ddim = ddim_steps is not None
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(
                    cond=mask_z,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            log["samples_image"] = self.decode_first_stage(samples)
        return log


class WormJointLatentDiffusion(LatentDiffusion):
    def __init__(self, *args, image_loss_weight=1.0, mask_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if image_loss_weight < 0.0 or mask_loss_weight < 0.0:
            raise ValueError("image_loss_weight and mask_loss_weight must be non-negative.")
        if image_loss_weight == 0.0 and mask_loss_weight == 0.0:
            raise ValueError("At least one modality loss weight must be positive.")
        self.image_loss_weight = image_loss_weight
        self.mask_loss_weight = mask_loss_weight

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

    def p_losses(self, x_start, cond, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        image_channels = self.first_stage_model.image_channels
        mask_channels = self.first_stage_model.mask_channels
        total_channels = image_channels + mask_channels

        image_model_output = model_output[:, :image_channels]
        image_target = target[:, :image_channels]
        mask_model_output = model_output[:, image_channels:total_channels]
        mask_target = target[:, image_channels:total_channels]

        loss_simple_image = self.get_loss(image_model_output, image_target, mean=False).mean([1, 2, 3])
        loss_simple_mask = self.get_loss(mask_model_output, mask_target, mean=False).mean([1, 2, 3])
        loss_simple_unweighted = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        image_fraction = image_channels / total_channels
        mask_fraction = mask_channels / total_channels
        loss_denominator = self.image_loss_weight * image_fraction + self.mask_loss_weight * mask_fraction
        loss_simple = (
            self.image_loss_weight * image_fraction * loss_simple_image
            + self.mask_loss_weight * mask_fraction * loss_simple_mask
        ) / loss_denominator

        loss_dict.update({
            f"{prefix}/loss_simple": loss_simple.mean(),
            f"{prefix}/loss_simple_unweighted": loss_simple_unweighted.mean(),
            f"{prefix}/loss_simple_image": loss_simple_image.mean(),
            f"{prefix}/loss_simple_mask": loss_simple_mask.mean(),
        })

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = (self.lvlb_weights[t] * loss_simple).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=100, ddim_eta=0.0,
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
