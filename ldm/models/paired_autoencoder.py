import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.models.autoencoder import AutoencoderKL


def _copy_ddconfig(ddconfig, in_channels=None, out_ch=None, double_z=None):
    config = copy.deepcopy(ddconfig)
    if in_channels is not None:
        config["in_channels"] = in_channels
    if out_ch is not None:
        config["out_ch"] = out_ch
    if double_z is not None:
        config["double_z"] = double_z
    return config


def _latent_size_from_ddconfig(ddconfig):
    return int(ddconfig["resolution"]) // (2 ** (len(ddconfig["ch_mult"]) - 1))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, context_dim=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head or dim // heads
        inner_dim = self.heads * self.dim_head
        context_dim = context_dim or dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        context = x if context is None else context
        b, n, _ = x.shape
        _, m, _ = context.shape
        h, d = self.heads, self.dim_head

        q = self.to_q(x).view(b, n, h, d).permute(0, 2, 1, 3)
        k = self.to_k(context).view(b, m, h, d).permute(0, 2, 1, 3)
        v = self.to_v(context).view(b, m, h, d).permute(0, 2, 1, 3)

        sim = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class PairedAutoencoderKLBase(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            embed_dim,
            image_key="image",
            mask_key="mask",
            monitor="val/rec_loss",
            kl_weight=1.0e-6,
            image_log_scale_init=0.0,
            mask_log_scale_init=0.0,
            learn_log_scales=True,
            mask_pos_weight=None,
            sample_posterior=True,
            ckpt_path=None,
            ignore_keys=None,
            **kwargs,
    ):
        super().__init__()
        self.ddconfig = ddconfig
        self.embed_dim = embed_dim
        self.image_key = image_key
        self.mask_key = mask_key
        self.monitor = monitor
        self.kl_weight = kl_weight
        self.mask_pos_weight = mask_pos_weight
        self.sample_posterior = sample_posterior
        self.learning_rate = None
        image_log_scale = torch.tensor(float(image_log_scale_init))
        mask_log_scale = torch.tensor(float(mask_log_scale_init))
        if learn_log_scales:
            self.image_log_scale = nn.Parameter(image_log_scale)
            self.mask_log_scale = nn.Parameter(mask_log_scale)
        else:
            self.register_buffer("image_log_scale", image_log_scale)
            self.register_buffer("mask_log_scale", mask_log_scale)

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
        image = batch[self.image_key]
        mask = batch[self.mask_key]
        if len(image.shape) == 3:
            image = image[..., None]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        mask = mask.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return image, mask

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

    def paired_loss(self, image, mask, outputs, posterior, split="train"):
        image_rec = outputs["image"]
        mask_logits = outputs["mask_logits"]
        target = self.mask_target(mask)

        image_l1 = F.l1_loss(image_rec, image)
        pos_weight = None
        if self.mask_pos_weight is not None:
            pos_weight = torch.tensor([self.mask_pos_weight], device=mask_logits.device, dtype=mask_logits.dtype)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pos_weight)
        mask_dice = self.dice_loss(mask_logits, target)
        kl_loss = posterior.kl().mean()

        image_nll = torch.exp(-self.image_log_scale) * image_l1 + self.image_log_scale
        mask_nll = torch.exp(-self.mask_log_scale) * mask_bce + self.mask_log_scale
        rec_loss = image_nll + mask_nll
        loss = rec_loss + self.kl_weight * kl_loss
        log = {
            "{}/total_loss".format(split): loss.detach(),
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/image_nll".format(split): image_nll.detach(),
            "{}/mask_nll".format(split): mask_nll.detach(),
            "{}/image_l1".format(split): image_l1.detach(),
            "{}/mask_bce".format(split): mask_bce.detach(),
            "{}/mask_dice_metric".format(split): mask_dice.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
            "{}/image_log_scale".format(split): self.image_log_scale.detach(),
            "{}/mask_log_scale".format(split): self.mask_log_scale.detach(),
        }
        return loss, log

    def encode(self, image, mask):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, image, mask, sample_posterior=None):
        sample_posterior = self.sample_posterior if sample_posterior is None else sample_posterior
        posterior = self.encode(image, mask)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def training_step(self, batch, batch_idx):
        image, mask = self.get_input(batch)
        batch_size = image.shape[0]
        outputs, posterior = self(image, mask)
        loss, log_dict = self.paired_loss(image, mask, outputs, posterior, split="train")
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
        image, mask = self.get_input(batch)
        batch_size = image.shape[0]
        outputs, posterior = self(image, mask, sample_posterior=False)
        loss, log_dict = self.paired_loss(image, mask, outputs, posterior, split="val")
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
        return torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9))

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        image, mask = self.get_input(batch)
        image = image.to(self.device)
        mask = mask.to(self.device)
        log["inputs_image"] = image
        log["inputs_mask"] = self.mask_to_rgb(mask)
        if only_inputs:
            return log

        outputs, posterior = self(image, mask, sample_posterior=False)
        log["reconstructions_image"] = outputs["image"]
        log["reconstructions_mask"] = self.mask_to_rgb(torch.sigmoid(outputs["mask_logits"]) * 2.0 - 1.0)

        samples = self.decode(torch.randn_like(posterior.mean))
        log["samples_image"] = samples["image"]
        log["samples_mask"] = self.mask_to_rgb(torch.sigmoid(samples["mask_logits"]) * 2.0 - 1.0)
        return log


class PairedAutoencoderKL(PairedAutoencoderKLBase):
    """Two modality-specific encoders with additive Gaussian-moment fusion."""

    def __init__(self, ddconfig, embed_dim, **kwargs):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim, **kwargs)
        z_channels = int(ddconfig["z_channels"])
        image_enc_config = _copy_ddconfig(ddconfig, in_channels=3, double_z=True)
        mask_enc_config = _copy_ddconfig(ddconfig, in_channels=1, double_z=True)
        image_dec_config = _copy_ddconfig(ddconfig, out_ch=3)
        mask_dec_config = _copy_ddconfig(ddconfig, out_ch=1)

        self.image_encoder = Encoder(**image_enc_config)
        self.mask_encoder = Encoder(**mask_enc_config)
        self.image_quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.mask_quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.image_decoder = Decoder(**image_dec_config)
        self.mask_decoder = Decoder(**mask_dec_config)
        self.init_pending_ckpt()

    def encode(self, image, mask):
        image_h = self.image_encoder(image)
        mask_h = self.mask_encoder(mask)
        moments = self.image_quant_conv(image_h) + self.mask_quant_conv(mask_h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        h = self.post_quant_conv(z)
        return {
            "image": self.image_decoder(h),
            "mask_logits": self.mask_decoder(h),
        }


class ConcatAutoencoderKL(PairedAutoencoderKLBase):
    """A 4-channel concatenation baseline over RGB image plus binary mask."""

    def __init__(self, ddconfig, embed_dim, **kwargs):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim, **kwargs)
        z_channels = int(ddconfig["z_channels"])
        enc_config = _copy_ddconfig(ddconfig, in_channels=4, double_z=True)
        dec_config = _copy_ddconfig(ddconfig, out_ch=4)

        self.encoder = Encoder(**enc_config)
        self.decoder = Decoder(**dec_config)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.init_pending_ckpt()

    def encode(self, image, mask):
        h = self.encoder(torch.cat([image, mask], dim=1))
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        h = self.post_quant_conv(z)
        rec = self.decoder(h)
        return {
            "image": rec[:, :3],
            "mask_logits": rec[:, 3:4],
        }


class WormLegacyJointAutoencoderKL(AutoencoderKL):
    """4-channel AutoencoderKL wrapper for legacy image+mask checkpoints."""

    def __init__(self, *args, log_binarized=True, log_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_binarized = log_binarized
        self.log_samples = log_samples

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    @staticmethod
    def split_joint(x):
        if x.shape[1] < 4:
            raise ValueError("Expected at least 4 channels for image+mask, got {}".format(tuple(x.shape)))
        return x[:, :3], x[:, 3:4]

    @staticmethod
    def mask_from_channel(mask_channel, binarized=True):
        if binarized:
            return torch.where(mask_channel > 0.0, torch.ones_like(mask_channel), -torch.ones_like(mask_channel))
        return torch.clamp(mask_channel, -1.0, 1.0)

    def batch_to_inputs(self, batch, device=None):
        x = self.get_input(batch, self.image_key)
        image, mask = self.split_joint(x)
        if device is not None:
            image = image.to(device)
            mask = mask.to(device)
        return image, mask

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        image, mask = self.split_joint(x)
        log["inputs_image"] = image
        log["inputs_mask"] = self.mask_to_rgb(self.mask_from_channel(mask, binarized=True))
        if only_inputs:
            return log

        reconstruction, posterior = self(x, sample_posterior=False)
        rec_image, rec_mask = self.split_joint(reconstruction)
        log["reconstructions_image"] = rec_image
        log["reconstructions_mask"] = self.mask_to_rgb(
            self.mask_from_channel(rec_mask, binarized=self.log_binarized)
        )
        if self.log_samples:
            samples = self.decode(torch.randn_like(posterior.mean))
            sample_image, sample_mask = self.split_joint(samples)
            log["samples_image"] = sample_image
            log["samples_mask"] = self.mask_to_rgb(self.mask_from_channel(sample_mask, binarized=True))
        return log


class WormJointAutoencoderKL(PairedAutoencoderKLBase):
    """Joint RGB+mask KL autoencoder with explicit modality reconstruction losses."""

    def __init__(
            self,
            ddconfig,
            embed_dim,
            image_l1_weight=1.0,
            perceptual_weight=0.0,
            foreground_image_weight=1.0,
            mask_bce_weight=1.0,
            mask_dice_weight=1.0,
            mask_boundary_weight=0.2,
            mask_pos_weight=None,
            log_binarized=True,
            log_samples=False,
            **kwargs,
    ):
        learn_log_scales = kwargs.pop("learn_log_scales", False)
        super().__init__(
            ddconfig=ddconfig,
            embed_dim=embed_dim,
            learn_log_scales=learn_log_scales,
            mask_pos_weight=mask_pos_weight,
            **kwargs,
        )
        z_channels = int(ddconfig["z_channels"])
        enc_config = _copy_ddconfig(ddconfig, in_channels=4, double_z=True)
        dec_config = _copy_ddconfig(ddconfig, out_ch=4)

        self.encoder = Encoder(**enc_config)
        self.decoder = Decoder(**dec_config)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.image_l1_weight = image_l1_weight
        self.perceptual_weight = perceptual_weight
        self.foreground_image_weight = foreground_image_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.mask_boundary_weight = mask_boundary_weight
        self.log_binarized = log_binarized
        self.log_samples = log_samples

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
                        "WormJointAutoencoderKL with perceptual_weight > 0 requires "
                        "the lpips package or taming-transformers. Install the repo "
                        "dependencies or set model.params.perceptual_weight=0.0."
                    ) from exc
                self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False

        self.init_pending_ckpt()

    @staticmethod
    def boundary_map(mask):
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return torch.clamp(dilated - eroded, 0.0, 1.0)

    @classmethod
    def boundary_loss(cls, logits, target):
        probs = torch.sigmoid(logits)
        return F.l1_loss(cls.boundary_map(probs), cls.boundary_map(target))

    @staticmethod
    def logits_to_mask(logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        return torch.where(probs > threshold, torch.ones_like(probs), -torch.ones_like(probs))

    def batch_to_inputs(self, batch, device=None):
        image, mask = self.get_input(batch)
        if device is not None:
            image = image.to(device)
            mask = mask.to(device)
        return image, mask

    def encode(self, image, mask):
        h = self.encoder(torch.cat([image, mask], dim=1))
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    @torch.no_grad()
    def encode_mode(self, image, mask):
        return self.encode(image, mask).mode()

    def decode(self, z):
        h = self.post_quant_conv(z)
        rec = self.decoder(h)
        return {
            "image": rec[:, :3],
            "mask_logits": rec[:, 3:4],
        }

    def paired_loss(self, image, mask, outputs, posterior, split="train"):
        image_rec = outputs["image"]
        mask_logits = outputs["mask_logits"]
        target = self.mask_target(mask)

        abs_image_error = torch.abs(image_rec - image)
        image_l1 = abs_image_error.mean()
        foreground_den = (target.sum() * image.shape[1]).clamp_min(1.0)
        foreground_image_l1 = (abs_image_error * target).sum() / foreground_den
        if self.perceptual_loss is None:
            perceptual_loss = torch.zeros((), device=image.device, dtype=image.dtype)
        else:
            perceptual_loss = self.perceptual_loss(image.contiguous(), image_rec.contiguous()).mean()

        pos_weight = None
        if self.mask_pos_weight is not None:
            pos_weight = torch.tensor([self.mask_pos_weight], device=mask_logits.device, dtype=mask_logits.dtype)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pos_weight)
        mask_dice = self.dice_loss(mask_logits, target)
        mask_boundary = (
            self.boundary_loss(mask_logits, target)
            if self.mask_boundary_weight > 0.0
            else torch.zeros((), device=image.device, dtype=image.dtype)
        )
        kl_loss = posterior.kl().mean()

        image_loss = (
            self.image_l1_weight * image_l1
            + self.foreground_image_weight * foreground_image_l1
            + self.perceptual_weight * perceptual_loss
        )
        mask_loss = (
            self.mask_bce_weight * mask_bce
            + self.mask_dice_weight * mask_dice
            + self.mask_boundary_weight * mask_boundary
        )
        rec_loss = image_loss + mask_loss
        loss = rec_loss + self.kl_weight * kl_loss

        with torch.no_grad():
            probs = torch.sigmoid(mask_logits)
            pred = (probs > 0.5).float()
            target_bin = (target > 0.5).float()
            intersection = torch.sum(pred * target_bin, dim=(1, 2, 3))
            union = torch.sum((pred + target_bin) > 0.0, dim=(1, 2, 3)).float()
            mask_iou = ((intersection + 1.0e-6) / (union + 1.0e-6)).mean()
            mask_pixel_acc = (pred == target_bin).float().mean()
            foreground_fraction = target_bin.mean()

        log = {
            "{}/total_loss".format(split): loss.detach(),
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/image_loss".format(split): image_loss.detach(),
            "{}/mask_loss".format(split): mask_loss.detach(),
            "{}/image_l1".format(split): image_l1.detach(),
            "{}/foreground_image_l1".format(split): foreground_image_l1.detach(),
            "{}/image_lpips".format(split): perceptual_loss.detach(),
            "{}/mask_bce".format(split): mask_bce.detach(),
            "{}/mask_dice".format(split): mask_dice.detach(),
            "{}/mask_boundary".format(split): mask_boundary.detach(),
            "{}/mask_iou".format(split): mask_iou.detach(),
            "{}/mask_pixel_acc".format(split): mask_pixel_acc.detach(),
            "{}/foreground_fraction".format(split): foreground_fraction.detach(),
            "{}/kl_loss".format(split): kl_loss.detach(),
        }
        return loss, log

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        image, mask = self.get_input(batch)
        image = image.to(self.device)
        mask = mask.to(self.device)
        log["inputs_image"] = image
        log["inputs_mask"] = self.mask_to_rgb(mask)
        if only_inputs:
            return log

        outputs, posterior = self(image, mask, sample_posterior=False)
        log["reconstructions_image"] = outputs["image"]
        if self.log_binarized:
            recon_mask = self.logits_to_mask(outputs["mask_logits"])
        else:
            recon_mask = torch.sigmoid(outputs["mask_logits"]) * 2.0 - 1.0
        log["reconstructions_mask"] = self.mask_to_rgb(recon_mask)

        if self.log_samples:
            samples = self.decode(torch.randn_like(posterior.mean))
            log["samples_image"] = samples["image"]
            log["samples_mask"] = self.mask_to_rgb(self.logits_to_mask(samples["mask_logits"]))
        return log


class ModalityTokenBottleneckAutoencoderKL(PairedAutoencoderKLBase):
    """Native paired AE using modality tokens and a shared latent bottleneck."""

    def __init__(
            self,
            ddconfig,
            embed_dim,
            token_dim=128,
            token_heads=4,
            token_dim_head=None,
            latent_blocks=2,
            ff_mult=4,
            token_dropout=0.0,
            **kwargs,
    ):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim, **kwargs)
        self.token_dim = token_dim
        self.latent_size = _latent_size_from_ddconfig(ddconfig)
        z_channels = int(ddconfig["z_channels"])

        image_enc_config = _copy_ddconfig(ddconfig, in_channels=3, double_z=False)
        mask_enc_config = _copy_ddconfig(ddconfig, in_channels=1, double_z=False)
        image_dec_config = _copy_ddconfig(ddconfig, out_ch=3)
        mask_dec_config = _copy_ddconfig(ddconfig, out_ch=1)

        self.image_encoder = Encoder(**image_enc_config)
        self.mask_encoder = Encoder(**mask_enc_config)
        self.image_proj = nn.Conv2d(z_channels, token_dim, 1)
        self.mask_proj = nn.Conv2d(z_channels, token_dim, 1)

        self.image_pos = nn.Parameter(torch.zeros(1, token_dim, self.latent_size, self.latent_size))
        self.mask_pos = nn.Parameter(torch.zeros(1, token_dim, self.latent_size, self.latent_size))
        self.latent_queries = nn.Parameter(torch.randn(1, token_dim, self.latent_size, self.latent_size) * 0.02)
        self.latent_pos = nn.Parameter(torch.zeros(1, token_dim, self.latent_size, self.latent_size))
        self.image_modality = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.mask_modality = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.norm_latent_img = nn.LayerNorm(token_dim)
        self.norm_latent_mask = nn.LayerNorm(token_dim)
        self.norm_image_tokens = nn.LayerNorm(token_dim)
        self.norm_mask_tokens = nn.LayerNorm(token_dim)
        self.cross_image = Attention(token_dim, heads=token_heads, dim_head=token_dim_head)
        self.cross_mask = Attention(token_dim, heads=token_heads, dim_head=token_dim_head)
        self.gate_image = nn.Parameter(torch.ones(1))
        self.gate_mask = nn.Parameter(torch.ones(1))
        self.latent_blocks = nn.ModuleList([
            TransformerBlock(
                token_dim,
                heads=token_heads,
                dim_head=token_dim_head,
                ff_mult=ff_mult,
                dropout=token_dropout,
            )
            for _ in range(latent_blocks)
        ])
        self.to_moments_norm = nn.LayerNorm(token_dim)
        self.to_moments = nn.Linear(token_dim, 2 * embed_dim)

        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.image_decoder = Decoder(**image_dec_config)
        self.mask_decoder = Decoder(**mask_dec_config)
        self.init_pending_ckpt()

    @staticmethod
    def _flatten_tokens(x):
        return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])

    @staticmethod
    def _unflatten_tokens(x, height, width):
        b, _, c = x.shape
        return x.view(b, height, width, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _resize_pos(pos, size):
        if pos.shape[-2:] == size:
            return pos
        return F.interpolate(pos, size=size, mode="bilinear", align_corners=False)

    def _tokens_from_feature(self, feature, projection, pos, modality):
        feature = projection(feature)
        feature = feature + self._resize_pos(pos, feature.shape[-2:])
        tokens = self._flatten_tokens(feature)
        return tokens + modality

    def encode(self, image, mask):
        image_tokens = self._tokens_from_feature(
            self.image_encoder(image), self.image_proj, self.image_pos, self.image_modality
        )
        mask_tokens = self._tokens_from_feature(
            self.mask_encoder(mask), self.mask_proj, self.mask_pos, self.mask_modality
        )

        latent = self.latent_queries + self.latent_pos
        latent = latent.expand(image.shape[0], -1, -1, -1)
        latent_tokens = self._flatten_tokens(latent)
        latent_tokens = latent_tokens + self.gate_image * self.cross_image(
            self.norm_latent_img(latent_tokens), self.norm_image_tokens(image_tokens)
        )
        latent_tokens = latent_tokens + self.gate_mask * self.cross_mask(
            self.norm_latent_mask(latent_tokens), self.norm_mask_tokens(mask_tokens)
        )
        for block in self.latent_blocks:
            latent_tokens = block(latent_tokens)

        moments = self.to_moments(self.to_moments_norm(latent_tokens))
        moments = self._unflatten_tokens(moments, self.latent_size, self.latent_size)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        h = self.post_quant_conv(z)
        return {
            "image": self.image_decoder(h),
            "mask_logits": self.mask_decoder(h),
        }
