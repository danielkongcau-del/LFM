import os
import math
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from datasets import CachedFolder
from models import make_vqmodel, EMA, MaskGITSampler
from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import get_time_str, check_freq, set_seed
from utils.experiment import create_exp_dir, find_resume_checkpoint, instantiate_from_config, discard_label
from utils.distributed import init_distributed_mode, is_main_process, on_main_process, is_dist_avail_and_initialized
from utils.distributed import wait_for_everyone, cleanup, gather_tensor, get_rank, get_world_size, get_local_rank, main_process_first


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-mp', '--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f'Process {get_rank()} using device: {device}', flush=True)
    wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if is_main_process():
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt', 'samples'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=is_main_process(),
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=is_main_process(),
    )

    # SET MIXED PRECISION
    if args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        mp_dtype = torch.bfloat16
    else:
        mp_dtype = torch.float32

    # SET SEED
    set_seed(conf.seed + get_rank())
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {get_world_size()}')
    logger.info(f'Distributed mode: {is_dist_avail_and_initialized()}')
    logger.info(f'Mixed precision: {args.mixed_precision}')
    wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % get_world_size() == 0
    bspp = conf.train.batch_size // get_world_size()  # batch size per process
    micro_batch_size = conf.train.micro_batch_size or bspp  # actual batch size in each iteration
    train_set = load_data(conf.data, split='all' if conf.data.name.lower() == 'ffhq' else 'train')
    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    train_loader = DataLoader(train_set, batch_size=bspp, sampler=train_sampler, drop_last=True, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Micro batch size: {micro_batch_size}')
    logger.info(f'Gradient accumulation steps: {math.ceil(bspp / micro_batch_size)}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # LOAD PRETRAINED VQMODEL
    with main_process_first():
        vqmodel = make_vqmodel(conf.vqmodel.model_name)
    vqmodel = vqmodel.requires_grad_(False).eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained vqmodel: {conf.vqmodel.model_name}')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.transformer).to(device)
    ema = EMA(model.parameters(), **getattr(conf.train, 'ema', dict()))
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision == 'fp16')
    logger.info(f'Number of parameters of transformer: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # BUILD SAMPLER
    fm_size = conf.data.img_size // vqmodel.downsample_factor  # feature map size
    sampler = MaskGITSampler(model=model, sequence_length=fm_size ** 2, sampling_steps=8, device=device)

    # RESUME TRAINING
    step, epoch = 0, 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load training states (ema, optimizer, scheduler, scaler, step, epoch)
        ckpt = torch.load(os.path.join(resume_path, 'training_states.pt'), map_location='cpu', weights_only=True)
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        step = ckpt['step'] + 1
        epoch = ckpt['epoch']
        logger.info(f'Successfully load training states from {resume_path}')
        logger.info(f'Restart training at step {step}')
        del ckpt

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())
    model_wo_ddp = model.module if is_dist_avail_and_initialized() else model
    ema.to(device)
    wait_for_everyone()

    # TRAINING FUNCTIONS
    @on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model and ema model
        torch.save(dict(model=model_wo_ddp.state_dict()), os.path.join(save_path, 'model.pt'))
        with ema.scope(model.parameters()):
            torch.save(dict(model=model_wo_ddp.state_dict()), os.path.join(save_path, 'model_ema.pt'))
        # save training states (ema, optimizer, scheduler, scaler, step, epoch)
        torch.save(dict(
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            step=step,
            epoch=epoch,
        ), os.path.join(save_path, 'training_states.pt'))

    def train_micro_batch(micro_batch, loss_scale, no_sync):
        idx = micro_batch
        B, L = idx.shape
        with model.no_sync() if no_sync else nullcontext():
            with torch.autocast(device_type='cuda', dtype=mp_dtype):
                # transformer forward
                mask = model_wo_ddp.get_random_mask(B, L)                           # (B, L)
                masked_idx = torch.where(mask, model_wo_ddp.mask_token_id, idx)     # (B, L)
                preds = model(masked_idx).reshape(B * L, -1)                        # (B * L, C)
                mask = mask.reshape(B * L)                                          # (B * L)
                # cross-entropy loss
                target = idx.reshape(-1)                                            # (B * L)
                target = torch.where(mask, target, -100)
                loss = F.cross_entropy(
                    input=preds, target=target, ignore_index=-100,
                    label_smoothing=conf.train.label_smoothing,
                )
                loss = loss * loss_scale
            # backward
            scaler.scale(loss).backward()
        return loss

    def train_step(batch):
        # get data
        if isinstance(train_set, CachedFolder):
            idx = batch['idx'].long().to(device)
            B, L = idx.shape
        else:
            x = discard_label(batch).float().to(device)
            B, N = x.shape[0], conf.data.img_size // vqmodel.downsample_factor
            L = N * N
            # vqmodel encode
            with torch.no_grad():
                idx = vqmodel.encode(x)['indices'].reshape(B, L)

        # zero the gradients
        optimizer.zero_grad()

        # forward and backward with gradient accumulation
        loss = torch.tensor(0., device=device)
        for i in range(0, B, micro_batch_size):
            idx_micro_batch = idx[i:i+micro_batch_size]
            loss_scale = idx_micro_batch.shape[0] / B
            no_sync = i + micro_batch_size < B and is_dist_avail_and_initialized()
            loss_micro_batch = train_micro_batch(idx_micro_batch, loss_scale, no_sync)
            loss = loss + loss_micro_batch

        # optimize
        if conf.train.get('clip_grad_norm', None):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.train.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model.parameters())
        scheduler.step()
        return dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def sample(savepath):
        n_samples = conf.train.n_samples // get_world_size()
        with ema.scope(model.parameters()):
            idx = sampler.sample(n_samples=n_samples)
        samples = vqmodel.decode_indices(idx, shape=(n_samples, fm_size, fm_size, -1)).clamp(-1, 1)
        samples = torch.cat(gather_tensor(samples), dim=0).cpu()
        if is_main_process():
            nrow = math.ceil(math.sqrt(conf.train.n_samples))
            save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    while step < conf.train.n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        for _batch in tqdm.tqdm(train_loader, desc='Epoch', leave=False, disable=not is_main_process()):
            # train a step
            model.train()
            train_status = train_step(_batch)
            status_tracker.track_status('Train', train_status, step)
            wait_for_everyone()
            # validate
            model.eval()
            # save checkpoint
            if check_freq(conf.train.save_freq, step):
                save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
                wait_for_everyone()
            # sample from current model
            if check_freq(conf.train.sample_freq, step):
                sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
                wait_for_everyone()
            step += 1
            if step >= conf.train.n_steps:
                break
        epoch += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    wait_for_everyone()

    # END OF TRAINING
    status_tracker.close()
    cleanup()
    logger.info('End of training')


if __name__ == '__main__':
    main()
