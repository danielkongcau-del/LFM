import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


def _copy_ddconfig(ddconfig, in_channels=None, out_ch=None, double_z=None):
    config = copy.deepcopy(ddconfig)
    if in_channels is not None:
        config["in_channels"] = in_channels
    if out_ch is not None:
        config["out_ch"] = out_ch
    if double_z is not None:
        config["double_z"] = double_z
    return config


class WormAutoencoderKLBase(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            embed_dim,
            input_key,
            monitor="val/rec_loss",
            kl_weight=1.0e-6,
            sample_posterior=True,
            ckpt_path=None,
            ignore_keys=None,
            **kwargs,
    ):
        super().__init__()
        self.ddconfig = ddconfig
        self.embed_dim = embed_dim
        self.input_key = input_key
        self.monitor = monitor
        self.kl_weight = kl_weight
        self.sample_posterior = sample_posterior
        self.learning_rate = None
        self._ckpt_path = ckpt_path
        self._ignore_keys = ignore_keys or []

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for key in keys:
            for ignore_key in ignore_keys:
                if key.startswith(ignore_key):
                    print("Deleting key {} from state_dict.".format(key))
                    del sd[key]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("Restored from {} with {} missing and {} unexpected keys".format(
            path, len(missing), len(unexpected)
        ))

    def init_pending_ckpt(self):
        if self._ckpt_path is not None:
            self.init_from_ckpt(self._ckpt_path, ignore_keys=self._ignore_keys)

    def get_input(self, batch):
        x = batch[self.input_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=None):
        sample_posterior = self.sample_posterior if sample_posterior is None else sample_posterior
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    @staticmethod
    def kl_loss(posterior):
        return posterior.kl().mean()

    def reconstruction_loss(self, x, reconstruction, posterior, split):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        batch_size = x.shape[0]
        reconstruction, posterior = self(x)
        loss, log_dict = self.reconstruction_loss(x, reconstruction, posterior, split="train")
        self.log(
            "aeloss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        batch_size = x.shape[0]
        reconstruction, posterior = self(x, sample_posterior=False)
        loss, log_dict = self.reconstruction_loss(x, reconstruction, posterior, split="val")
        self.log(
            "val/rec_loss",
            log_dict["val/rec_loss"],
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log_dict(
            {k: v for k, v in log_dict.items() if k != "val/rec_loss"},
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class WormImageAutoencoderKL(WormAutoencoderKLBase):
    """Image-only KL autoencoder with an original-LDM-style L1 + LPIPS preset."""

    def __init__(
            self,
            ddconfig,
            embed_dim,
            image_key="image",
            l1_weight=1.0,
            perceptual_weight=1.0,
            **kwargs,
    ):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim, input_key=image_key, **kwargs)
        z_channels = int(ddconfig["z_channels"])
        enc_config = _copy_ddconfig(ddconfig, in_channels=3, double_z=True)
        dec_config = _copy_ddconfig(ddconfig, out_ch=3)

        self.encoder = Encoder(**enc_config)
        self.decoder = Decoder(**dec_config)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = None
        if self.perceptual_weight > 0.0:
            try:
                import lpips
                self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
            except ImportError as exc:
                try:
                    from taming.modules.losses.lpips import LPIPS
                except ImportError:
                    raise ImportError(
                        "WormImageAutoencoderKL with perceptual_weight > 0 requires "
                        "the lpips package or taming-transformers. Install the repo "
                        "dependencies or set model.params.perceptual_weight=0.0 for "
                        "an L1-only image AE."
                    ) from exc
                self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
        self.init_pending_ckpt()

    def reconstruction_loss(self, x, reconstruction, posterior, split):
        l1_loss = F.l1_loss(reconstruction, x)
        if self.perceptual_loss is None:
            perceptual_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            perceptual_loss = self.perceptual_loss(x.contiguous(), reconstruction.contiguous()).mean()
        kl_loss = self.kl_loss(posterior)
        rec_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss
        loss = rec_loss + self.kl_weight * kl_loss
        log = {
            "{}/total_loss".format(split): loss.detach(),
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/image_l1".format(split): l1_loss.detach(),
            "{}/image_lpips".format(split): perceptual_loss.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
        }
        return loss, log

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch).to(self.device)
        log["inputs_image"] = x
        if only_inputs:
            return log

        reconstruction, posterior = self(x, sample_posterior=False)
        log["reconstructions_image"] = reconstruction
        log["samples_image"] = self.decode(torch.randn_like(posterior.mean))
        return log


class WormMaskAutoencoderKL(WormAutoencoderKLBase):
    """Mask-only KL autoencoder using binary logits, BCE, and Dice loss."""

    def __init__(
            self,
            ddconfig,
            embed_dim,
            mask_key="mask",
            bce_weight=1.0,
            dice_weight=1.0,
            mask_pos_weight=None,
            **kwargs,
    ):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim, input_key=mask_key, **kwargs)
        z_channels = int(ddconfig["z_channels"])
        enc_config = _copy_ddconfig(ddconfig, in_channels=1, double_z=True)
        dec_config = _copy_ddconfig(ddconfig, out_ch=1)

        self.encoder = Encoder(**enc_config)
        self.decoder = Decoder(**dec_config)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.mask_pos_weight = mask_pos_weight
        self.init_pending_ckpt()

    @staticmethod
    def mask_target(mask):
        return torch.clamp((mask + 1.0) * 0.5, 0.0, 1.0)

    @staticmethod
    def dice_loss(logits, target, eps=1.0e-6):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    def reconstruction_loss(self, x, reconstruction, posterior, split):
        target = self.mask_target(x)
        pos_weight = None
        if self.mask_pos_weight is not None:
            pos_weight = torch.tensor([self.mask_pos_weight], device=x.device, dtype=x.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(reconstruction, target, pos_weight=pos_weight)
        dice_loss = self.dice_loss(reconstruction, target)
        kl_loss = self.kl_loss(posterior)
        rec_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        loss = rec_loss + self.kl_weight * kl_loss
        log = {
            "{}/total_loss".format(split): loss.detach(),
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/mask_bce".format(split): bce_loss.detach(),
            "{}/mask_dice".format(split): dice_loss.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
        }
        return loss, log

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch).to(self.device)
        log["inputs_mask"] = self.mask_to_rgb(x)
        if only_inputs:
            return log

        logits, posterior = self(x, sample_posterior=False)
        log["reconstructions_mask"] = self.mask_to_rgb(torch.sigmoid(logits) * 2.0 - 1.0)
        samples = self.decode(torch.randn_like(posterior.mean))
        log["samples_mask"] = self.mask_to_rgb(torch.sigmoid(samples) * 2.0 - 1.0)
        return log
