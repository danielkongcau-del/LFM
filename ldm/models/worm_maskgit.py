import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ldm.util import instantiate_from_config


def trusted_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_lightning_model(config_path, checkpoint_path, device):
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    payload = trusted_torch_load(checkpoint_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded {} with {} missing and {} unexpected keys".format(
        checkpoint_path, len(missing), len(unexpected)
    ))
    model.to(device)
    model.eval()
    return model


def get_mask_scheduling_fn(mask_schedule_type):
    if mask_schedule_type == "linear":
        return lambda r: 1.0 - r
    if mask_schedule_type == "cosine":
        return lambda r: np.cos(r * np.pi / 2.0)
    if mask_schedule_type == "arccos":
        return lambda r: np.arccos(r) / (np.pi / 2.0)
    if mask_schedule_type.startswith("pow"):
        exponent = float(mask_schedule_type[3:])
        return lambda r: 1.0 - r ** exponent
    raise ValueError("Unknown mask schedule type: {}".format(mask_schedule_type))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads.")
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, dim // self.n_heads).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, dim // self.n_heads).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, dim // self.n_heads).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scale = q.shape[-1] ** -0.5
            attn = torch.matmul(q * scale, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            x = torch.matmul(attn, v)

        x = x.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out_dropout(self.out(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MaskTransformer(nn.Module):
    """Bidirectional token transformer adapted from MaskGIT stage-2 training."""

    def __init__(
            self,
            vocab_size,
            embed_dim,
            n_heads,
            n_layers,
            n_tokens,
            dropout=0.0,
            mask_schedule_type="cosine",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_tokens = n_tokens
        self.gamma = get_mask_scheduling_fn(mask_schedule_type)
        self.mask_token_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros((1, n_tokens, embed_dim)))
        self.drop_emb = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        _, length = idx.shape
        if length > self.n_tokens:
            raise ValueError("Sequence length {} exceeds configured n_tokens {}.".format(length, self.n_tokens))
        x = self.token_emb(idx)
        x = x + self.pos_emb[:, :length, :]
        x = self.drop_emb(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    def get_random_mask(self, batch_size, length):
        device = self.pos_emb.device
        n = math.ceil(self.gamma(np.random.random()) * length)
        n = max(1, min(length, n))
        index = torch.rand((batch_size, length), device=device).topk(n, dim=1).indices
        mask = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=index, src=torch.ones_like(mask, dtype=torch.bool))
        return mask


class MaskGITSampler:
    def __init__(
            self,
            model,
            sequence_length,
            sampling_steps,
            softmax_temp=1.0,
            topk=None,
            base_gumbel_temp=4.5,
            token_logits_bias=None,
            token_logits_alpha=0.0,
            token_confidence_bias=None,
            token_confidence_alpha=0.0,
            device=None,
    ):
        if sampling_steps > sequence_length:
            raise ValueError("sampling_steps must be <= sequence_length.")
        self.model = model
        self.mask_token_id = model.mask_token_id
        self.sequence_length = sequence_length
        self.sampling_steps = sampling_steps
        self.softmax_temp = softmax_temp
        self.topk = topk
        self.base_gumbel_temp = base_gumbel_temp
        self.token_logits_bias = token_logits_bias
        self.token_logits_alpha = token_logits_alpha
        self.token_confidence_bias = token_confidence_bias
        self.token_confidence_alpha = token_confidence_alpha
        self.gumbel = torch.distributions.Gumbel(0, 1)
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def get_model_prediction(self, idx, token_logits_scale=0.0):
        logits = self.model(idx)
        if self.token_logits_bias is not None and token_logits_scale > 0.0:
            logits = logits + token_logits_scale * self.token_logits_bias.view(1, 1, -1)
        if self.topk is not None:
            cutoff, _ = torch.topk(logits, min(self.topk, logits.shape[-1]), dim=-1)
            logits = logits.masked_fill(logits < cutoff[..., [-1]], float("-inf"))
        return torch.softmax(logits / self.softmax_temp, dim=-1)

    @torch.no_grad()
    def sample_one_step(self, idx, n, gumbel_temp=0.0, token_logits_scale=0.0, token_confidence_scale=0.0):
        batch, length = idx.shape
        mask = idx == self.mask_token_id
        probs = self.get_model_prediction(idx, token_logits_scale=token_logits_scale)
        sampled_idx = torch.multinomial(probs.reshape(batch * length, -1), num_samples=1).reshape(batch, length)
        sampled_probs = torch.gather(probs, dim=-1, index=sampled_idx[:, :, None]).reshape(batch, length)

        sampled_idx = torch.where(mask, sampled_idx, idx)
        sampled_probs = torch.where(mask, sampled_probs, torch.full_like(sampled_probs, torch.inf))

        randomness = self.gumbel.sample(sampled_probs.shape).to(sampled_probs.device)
        confidence = torch.log(sampled_probs.clamp_min(1.0e-20)) + gumbel_temp * randomness
        if self.token_confidence_bias is not None and token_confidence_scale > 0.0:
            confidence_bias = torch.gather(
                self.token_confidence_bias.view(1, -1).expand(batch, -1),
                dim=1,
                index=sampled_idx.clamp_max(self.token_confidence_bias.numel() - 1),
            )
            confidence = confidence + token_confidence_scale * confidence_bias
        index = confidence.topk(length - n, dim=1).indices
        mask = mask.scatter(dim=1, index=index, src=torch.zeros_like(mask, dtype=torch.bool))
        sampled_idx = torch.where(mask, self.mask_token_id, sampled_idx)
        return sampled_idx, mask

    @torch.no_grad()
    def sample_loop(self, n_samples):
        batch, length, steps = n_samples, self.sequence_length, self.sampling_steps
        idx = torch.full((batch, length), self.mask_token_id, dtype=torch.long, device=self.device)
        for t in range(steps):
            n = math.floor(self.model.gamma((t + 1) / steps) * length)
            n = min(n, length - 1 - t)
            n = max(0, n)
            gumbel_temp = self.base_gumbel_temp * (1.0 - (t + 1) / steps)
            decay = 1.0 - t / max(steps - 1, 1)
            token_logits_scale = self.token_logits_alpha * decay
            token_confidence_scale = self.token_confidence_alpha * decay
            idx, mask = self.sample_one_step(
                idx,
                n=n,
                gumbel_temp=gumbel_temp,
                token_logits_scale=token_logits_scale,
                token_confidence_scale=token_confidence_scale,
            )
            yield idx, mask

    @torch.no_grad()
    def sample(self, n_samples):
        last = None
        for last in self.sample_loop(n_samples):
            pass
        return last[0]


class WormMaskTokenPrior(pl.LightningModule):
    def __init__(
            self,
            vocab_size,
            grid_size,
            embed_dim=256,
            n_heads=8,
            n_layers=8,
            dropout=0.1,
            mask_schedule_type="cosine",
            label_smoothing=0.0,
            weight_decay=0.01,
            sample_steps=12,
            sample_topk=None,
            sample_softmax_temp=1.0,
            sample_gumbel_temp=4.5,
            sample_token_weight_alpha=0.0,
            sample_token_logits_alpha=None,
            sample_token_confidence_alpha=0.0,
            tokenizer_config=None,
            tokenizer_ckpt=None,
            token_weight_path=None,
            token_weight_power=0.5,
            token_weight_min=0.25,
            token_weight_max=4.0,
            log_reconstruction_mask_ratio=0.5,
            monitor="val/loss",
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.n_tokens = grid_size * grid_size
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        self.sample_steps = sample_steps
        self.sample_topk = sample_topk
        self.sample_softmax_temp = sample_softmax_temp
        self.sample_gumbel_temp = sample_gumbel_temp
        self.sample_token_weight_alpha = sample_token_weight_alpha
        self.sample_token_logits_alpha = (
            sample_token_weight_alpha if sample_token_logits_alpha is None else sample_token_logits_alpha
        )
        self.sample_token_confidence_alpha = sample_token_confidence_alpha
        self.token_weight_path = token_weight_path
        self.token_weight_power = token_weight_power
        self.token_weight_min = token_weight_min
        self.token_weight_max = token_weight_max
        self.log_reconstruction_mask_ratio = log_reconstruction_mask_ratio
        self.monitor = monitor
        self.learning_rate = None

        self.transformer = MaskTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_tokens=self.n_tokens,
            dropout=dropout,
            mask_schedule_type=mask_schedule_type,
        )
        self.tokenizer_config = tokenizer_config
        self.tokenizer_ckpt = tokenizer_ckpt
        self.__dict__["_tokenizer"] = None
        self.__dict__["_token_weights"] = None
        if tokenizer_config is not None and tokenizer_ckpt is not None:
            self._load_tokenizer(device=torch.device("cpu"))

    def _load_tokenizer(self, device):
        tokenizer = self.__dict__.get("_tokenizer")
        if tokenizer is None:
            tokenizer = load_lightning_model(self.tokenizer_config, self.tokenizer_ckpt, device)
            tokenizer.requires_grad_(False)
            tokenizer.eval()
            self.__dict__["_tokenizer"] = tokenizer
        else:
            tokenizer.to(device)
        return tokenizer

    def _weights_from_counts(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.float32).view(-1)
        used = counts > 0
        weights = torch.zeros_like(counts)
        if used.any():
            used_counts = counts[used]
            weights[used] = (used_counts.mean() / used_counts).pow(self.token_weight_power)
            weights[used] = weights[used].clamp(self.token_weight_min, self.token_weight_max)
            weights[used] = weights[used] / weights[used].mean().clamp_min(1.0e-12)
        return weights

    def _load_token_weights(self, device):
        if not self.token_weight_path:
            return None

        weights = self.__dict__.get("_token_weights")
        if weights is None:
            payload = trusted_torch_load(self.token_weight_path, map_location="cpu")
            if isinstance(payload, dict):
                if "weights" in payload:
                    weights = payload["weights"]
                elif "counts" in payload:
                    weights = self._weights_from_counts(payload["counts"])
                else:
                    raise ValueError(
                        "Token weight file must contain `weights` or `counts`, got keys {}.".format(
                            sorted(payload.keys())
                        )
                    )
            else:
                weights = payload

            weights = torch.as_tensor(weights, dtype=torch.float32).view(-1)
            if weights.numel() != self.vocab_size:
                raise ValueError(
                    "Expected {} token weights, got {} from {}.".format(
                        self.vocab_size, weights.numel(), self.token_weight_path
                    )
                )
            self.__dict__["_token_weights"] = weights

        return weights.to(device)

    def on_fit_start(self):
        self._load_token_weights(self.device)
        if self.tokenizer_config is not None and self.tokenizer_ckpt is not None:
            self._load_tokenizer(self.device)

    def on_validation_start(self):
        tokenizer = self.__dict__.get("_tokenizer")
        if tokenizer is not None:
            tokenizer.to(self.device)

    @contextmanager
    def tokenizer_ready(self):
        if self.tokenizer_config is None or self.tokenizer_ckpt is None:
            yield None
            return
        tokenizer = self._load_tokenizer(self.device)
        was_training = tokenizer.training
        tokenizer.eval()
        try:
            yield tokenizer
        finally:
            if was_training:
                tokenizer.train()

    @staticmethod
    def get_input(batch):
        idx = batch["idx"] if isinstance(batch, dict) else batch
        return idx.long().view(idx.shape[0], -1)

    def forward(self, idx):
        return self.transformer(idx)

    def shared_step(self, batch, split):
        idx = self.get_input(batch).to(self.device)
        batch_size, length = idx.shape
        if length != self.n_tokens:
            raise ValueError("Expected {} tokens, got {}.".format(self.n_tokens, length))

        mask = self.transformer.get_random_mask(batch_size, length)
        masked_idx = torch.where(mask, self.transformer.mask_token_id, idx)
        logits = self(masked_idx).reshape(batch_size * length, -1)
        target = idx.reshape(-1)
        target = torch.where(mask.reshape(-1), target, torch.full_like(target, -100))
        token_weights = self._load_token_weights(idx.device)
        loss = F.cross_entropy(
            logits,
            target,
            weight=token_weights,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

        with torch.no_grad():
            valid = target != -100
            pred = logits.argmax(dim=-1)
            token_acc = (pred[valid] == target[valid]).float().mean() if valid.any() else torch.zeros((), device=idx.device)
            mask_ratio = mask.float().mean()

        log = {
            "{}/loss".format(split): loss.detach(),
            "{}/token_acc".format(split): token_acc.detach(),
            "{}/mask_ratio".format(split): mask_ratio.detach(),
        }
        return loss, log

    def training_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, split="train")
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, split="val")
        self.log("val/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({k: v for k, v in log.items() if k != "val/loss"}, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        params = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(params, lr=self.learning_rate, betas=(0.9, 0.95))

    @torch.no_grad()
    def sample_indices(self, batch_size, sampling_steps=None, topk=None, softmax_temp=None, gumbel_temp=None):
        token_bias = None
        if self.sample_token_logits_alpha > 0.0 or self.sample_token_confidence_alpha > 0.0:
            token_weights = self._load_token_weights(self.device)
            if token_weights is not None:
                token_bias = torch.log(token_weights.clamp_min(1.0e-6))
        sampler = MaskGITSampler(
            model=self.transformer,
            sequence_length=self.n_tokens,
            sampling_steps=sampling_steps or self.sample_steps,
            softmax_temp=softmax_temp or self.sample_softmax_temp,
            topk=self.sample_topk if topk is None else topk,
            base_gumbel_temp=self.sample_gumbel_temp if gumbel_temp is None else gumbel_temp,
            token_logits_bias=token_bias,
            token_logits_alpha=self.sample_token_logits_alpha,
            token_confidence_bias=token_bias,
            token_confidence_alpha=self.sample_token_confidence_alpha,
            device=self.device,
        )
        return sampler.sample(batch_size)

    @torch.no_grad()
    def masked_reconstruct_indices(self, idx, mask_ratio=None):
        mask_ratio = self.log_reconstruction_mask_ratio if mask_ratio is None else mask_ratio
        mask = torch.rand(idx.shape, device=idx.device) < mask_ratio
        masked_idx = torch.where(mask, self.transformer.mask_token_id, idx)
        logits = self(masked_idx)
        pred_idx = logits.argmax(dim=-1)
        return torch.where(mask, pred_idx, idx)

    @staticmethod
    def mask_to_rgb(mask):
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return mask.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def decode_indices_to_mask(self, indices, tokenizer):
        indices = indices.view(indices.shape[0], self.grid_size, self.grid_size)
        logits = tokenizer.decode_code(indices)
        probs = torch.sigmoid(logits)
        return torch.where(probs > 0.5, torch.ones_like(probs), -torch.ones_like(probs))

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        with self.tokenizer_ready() as tokenizer:
            if tokenizer is None:
                return log

            idx = self.get_input(batch).to(self.device)
            input_count = min(idx.shape[0], 4)
            log["inputs_mask"] = self.mask_to_rgb(self.decode_indices_to_mask(idx[:input_count], tokenizer))
            if only_inputs:
                return log

            reconstructed_idx = self.masked_reconstruct_indices(idx[:input_count])
            log["reconstructions_mask"] = self.mask_to_rgb(
                self.decode_indices_to_mask(reconstructed_idx, tokenizer)
            )
            sampled_idx = self.sample_indices(input_count)
            log["samples_mask"] = self.mask_to_rgb(self.decode_indices_to_mask(sampled_idx, tokenizer))
        return log
