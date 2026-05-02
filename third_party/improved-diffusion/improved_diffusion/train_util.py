import copy
import functools
import os
import re
import shutil

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        keep_latest=False,
        sample_diffusion=None,
        sample_interval=0,
        sample_num_samples=16,
        sample_batch_size=8,
        sample_image_size=256,
        sample_use_ddim=True,
        sample_clip_denoised=True,
        sample_dir="",
        sample_keep_latest=False,
        sample_save_raw=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.keep_latest = keep_latest
        self.sample_diffusion = sample_diffusion
        self.sample_interval = sample_interval
        self.sample_num_samples = sample_num_samples
        self.sample_batch_size = sample_batch_size
        self.sample_image_size = sample_image_size
        self.sample_use_ddim = sample_use_ddim
        self.sample_clip_denoised = sample_clip_denoised
        self.sample_dir = sample_dir
        self.sample_keep_latest = sample_keep_latest
        self.sample_save_raw = sample_save_raw

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if (
                self.sample_interval
                and self.step + self.resume_step > 0
                and (self.step + self.resume_step) % self.sample_interval == 0
            ):
                self.sample_latest()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        keep_filenames = []

        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                keep_filenames.append(filename)
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
            keep_filenames.append(opt_filename)
            with bf.BlobFile(
                bf.join(get_blob_logdir(), opt_filename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
            if self.keep_latest:
                self._cleanup_old_checkpoints(keep_filenames)

        dist.barrier()

    def sample_latest(self):
        if self.sample_diffusion is None or self.sample_num_samples <= 0:
            return
        if dist.get_rank() == 0:
            logger.log(
                "sampling {} masks at step {}...".format(
                    self.sample_num_samples, self.step + self.resume_step
                )
            )
            self._sample_latest_rank0()
        dist.barrier()

    def _sample_latest_rank0(self):
        outdir = self.sample_dir or os.path.join(logger.get_dir(), "samples")
        os.makedirs(outdir, exist_ok=True)
        if self.sample_keep_latest:
            self._cleanup_sample_dir(outdir)
        raw_dir = os.path.join(outdir, "raw")
        if self.sample_save_raw:
            os.makedirs(raw_dir, exist_ok=True)

        sample_fn = (
            self.sample_diffusion.ddim_sample_loop
            if self.sample_use_ddim
            else self.sample_diffusion.p_sample_loop
        )
        current_params = [param.detach().clone() for param in self.model_params]
        was_training = self.model.training
        masks = []

        try:
            ema_params = self.ema_params[0] if self.ema_params else self.master_params
            self._copy_params_to_model(ema_params)
            self.model.eval()

            written = 0
            while written < self.sample_num_samples:
                batch_size = min(
                    max(1, self.sample_batch_size),
                    self.sample_num_samples - written,
                )
                sample = sample_fn(
                    self.model,
                    (batch_size, 3, self.sample_image_size, self.sample_image_size),
                    clip_denoised=self.sample_clip_denoised,
                    model_kwargs={},
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()

                for i, raw in enumerate(sample):
                    index = written + i
                    gray = raw.mean(axis=2)
                    mask = (gray > 127.5).astype(np.uint8) * 255
                    Image.fromarray(mask, mode="L").save(
                        os.path.join(outdir, "{:06d}_mask.png".format(index))
                    )
                    masks.append(mask)
                    if self.sample_save_raw:
                        Image.fromarray(raw, mode="RGB").save(
                            os.path.join(raw_dir, "{:06d}_raw.png".format(index))
                        )
                written += batch_size

            if masks:
                self._save_sample_grid(masks, os.path.join(outdir, "grid.png"))
            logger.log("samples written to {}".format(outdir))
        finally:
            for param, saved in zip(self.model_params, current_params):
                param.detach().copy_(saved)
            if was_training:
                self.model.train()

    @staticmethod
    def _cleanup_sample_dir(outdir):
        raw_dir = os.path.join(outdir, "raw")
        if os.path.isdir(raw_dir):
            shutil.rmtree(raw_dir)
        for name in os.listdir(outdir):
            path = os.path.join(outdir, name)
            if name == "grid.png" or name.endswith("_mask.png"):
                os.remove(path)

    def _copy_params_to_model(self, params):
        if self.use_fp16:
            master_params_to_model_params(self.model_params, params)
            return
        for param, source in zip(self.model_params, params):
            param.detach().copy_(source.detach())

    @staticmethod
    def _save_sample_grid(masks, out_path):
        columns = int(np.ceil(np.sqrt(len(masks))))
        rows = int(np.ceil(len(masks) / columns))
        height, width = masks[0].shape
        grid = Image.new("L", (columns * width, rows * height), 0)
        for index, mask in enumerate(masks):
            x = (index % columns) * width
            y = (index // columns) * height
            grid.paste(Image.fromarray(mask, mode="L"), (x, y))
        grid.save(out_path)

    @staticmethod
    def _is_checkpoint_filename(name):
        return (
            re.fullmatch(r"model\d+\.pt", name) is not None
            or re.fullmatch(r"ema_[^_]+_\d+\.pt", name) is not None
            or re.fullmatch(r"opt\d+\.pt", name) is not None
        )

    def _cleanup_old_checkpoints(self, keep_filenames):
        logdir = get_blob_logdir()
        keep_filenames = set(keep_filenames)
        for name in bf.listdir(logdir):
            if name in keep_filenames or not self._is_checkpoint_filename(name):
                continue
            path = bf.join(logdir, name)
            logger.log(f"removing old checkpoint: {path}")
            bf.remove(path)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
