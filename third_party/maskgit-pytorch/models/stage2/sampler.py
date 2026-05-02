import math
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class BaseSampler:
    def __init__(
            self,
            model: nn.Module,
            sequence_length: int,
            sampling_steps: int,
            softmax_temp: float = 1.0,
            topk: int = None,
            cfg: float = 1.0,
            cfg_schedule: str = 'linear',
            device: torch.device = None,
    ):
        assert sampling_steps <= sequence_length
        assert cfg_schedule in ['constant', 'linear', 'linear-r'] or cfg_schedule.startswith('power-cosine-')
        self.model = model
        self.mask_token_id = self.model.mask_token_id
        self.sequence_length = sequence_length
        self.sampling_steps = sampling_steps
        self.softmax_temp = softmax_temp
        self.topk = topk
        self.cfg = cfg
        self.cfg_schedule = cfg_schedule
        self.device = device or next(model.parameters()).device

    def get_current_cfg(self, n: int, L: int, t: int, T: int):
        if self.cfg_schedule == 'constant':
            cfg_current = self.cfg
        elif self.cfg_schedule == 'linear':
            cfg_current = 1 + (self.cfg - 1) * (t / T)
        elif self.cfg_schedule == 'linear-r':
            cfg_current = 1 + (self.cfg - 1) * (L - n) / L
        elif self.cfg_schedule.startswith('power-cosine-'):
            power = float(self.cfg_schedule[13:])
            coef = (1 - np.cos(np.pi * ((t / T) ** power))) / 2
            cfg_current = 1 + (self.cfg - 1) * coef
        else:
            raise ValueError(f'Unknown cfg_schedule: {self.cfg_schedule}')
        return cfg_current

    @torch.no_grad()
    def get_model_prediction(self, idx: Tensor, y: Tensor = None, cfg: float = 1.0):
        L = idx.shape[1]
        logits = self.model(idx, y)
        if y is not None and cfg != 1.0:
            logits_uncond = self.model(idx, y, cond_drop_prob=1.0)
            logits = cfg * logits + (1 - cfg) * logits_uncond
        if self.topk is not None:
            v, _ = torch.topk(logits, min(self.topk, L), largest=True, sorted=True)
            logits[logits < v[..., [-1]]] = float('-inf')
        probs = torch.softmax(logits / self.softmax_temp, dim=-1)
        return probs


class MaskGITSampler(BaseSampler):
    def __init__(self, base_gumbel_temp: float = 4.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_gumbel_temp = base_gumbel_temp
        self.gumbel = torch.distributions.Gumbel(0, 1)

    def sample_one_step(self, idx: Tensor, n: int, y: Tensor = None, cfg: float = 1.0, gumbel_temp: float = 0.0):
        B, L = idx.shape
        mask = torch.eq(idx, self.mask_token_id)
        # get probabilities
        probs = self.get_model_prediction(idx, y, cfg)
        # sample all positions
        sampled_idx = torch.multinomial(probs.reshape(B * L, -1), num_samples=1).reshape(B, L)
        sampled_probs = torch.gather(probs, dim=-1, index=sampled_idx[:, :, None]).reshape(B, L)
        # restore unmasked positions
        sampled_idx = torch.where(mask, sampled_idx, idx)
        sampled_probs = torch.where(mask, sampled_probs, torch.full_like(sampled_probs, torch.inf))
        # unmask top L-n positions (with gumbel noise added for randomness)
        randomness = self.gumbel.sample(sampled_probs.shape).to(sampled_probs.device)
        confidence = torch.log(sampled_probs) + gumbel_temp * randomness
        index = confidence.topk(L - n, dim=1).indices
        mask = mask.scatter(dim=1, index=index, src=torch.zeros_like(mask, dtype=torch.bool))
        sampled_idx = torch.where(mask, self.mask_token_id, sampled_idx)
        return sampled_idx, mask

    def sample_loop(self, n_samples: int, y: Tensor = None):
        B, L, T = n_samples, self.sequence_length, self.sampling_steps
        idx = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=self.device)
        for t in range(T):
            # after this iteration, n positions remain masked
            n = math.floor(self.model.gamma((t + 1) / T) * L)
            n = min(n, L - 1 - t)
            cfg_current = self.get_current_cfg(n, L, t, T)
            gumbel_temp = self.base_gumbel_temp * (1 - (t + 1) / T)
            idx, mask = self.sample_one_step(idx, n, y, cfg_current, gumbel_temp)
            yield idx, mask

    def sample(self, n_samples: int, y: Tensor = None):
        *_, (idx, mask) = self.sample_loop(n_samples, y)
        return idx


class RandomSampler(BaseSampler):
    def sample_one_step(self, idx: Tensor, n: int, y: Tensor = None, cfg: float = 1.0):
        B, L = idx.shape
        mask = torch.eq(idx, self.mask_token_id)
        # get probabilities
        probs = self.get_model_prediction(idx, y, cfg)
        # sample all positions
        sampled_idx = torch.multinomial(probs.reshape(B * L, -1), num_samples=1).reshape(B, L)
        # restore unmasked positions
        sampled_idx = torch.where(mask, sampled_idx, idx)
        # preserve L-n positions (randomly selected)
        confidence = torch.rand_like(idx, dtype=torch.float)  # random confidence
        confidence = torch.where(mask, confidence, torch.full_like(confidence, torch.inf))
        index = confidence.topk(L - n, dim=1).indices
        mask = mask.scatter(dim=1, index=index, src=torch.zeros_like(mask, dtype=torch.bool))
        sampled_idx = torch.where(mask, self.mask_token_id, sampled_idx)
        return sampled_idx, mask

    def sample_loop(self, n_samples: int, y: Tensor = None):
        B, L, T = n_samples, self.sequence_length, self.sampling_steps
        idx = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=self.device)
        for t in range(T):
            # after this iteration, n positions remain masked
            n = math.floor(self.model.gamma((t + 1) / T) * L)
            n = min(n, L - 1 - t)
            cfg_current = self.get_current_cfg(n, L, t, T)
            idx, mask = self.sample_one_step(idx, n, y, cfg_current)
            yield idx, mask

    def sample(self, n_samples: int, y: Tensor = None):
        *_, (idx, mask) = self.sample_loop(n_samples, y)
        return idx
