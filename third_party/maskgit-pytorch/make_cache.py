import os
import tqdm
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.backends.cudnn.benchmark = True

from models import make_vqmodel
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import set_seed
from utils.distributed import init_distributed_mode, is_main_process, is_dist_avail_and_initialized
from utils.distributed import get_rank, get_world_size, wait_for_everyone, main_process_first, cleanup, gather_tensor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Set random seed')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory caching latents')
    parser.add_argument('--bspp', type=int, default=128, help='Batch size on each process')
    parser.add_argument('--full', action='store_true', default=False, help='Save full latents')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f'Process {get_rank()} using device: {device}', flush=True)
    wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=is_main_process())

    # SET SEED
    set_seed(args.seed + get_rank())
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {get_world_size()}')
    logger.info(f'Distributed mode: {is_dist_avail_and_initialized()}')
    wait_for_everyone()

    # BUILD DATASET & DATALOADER
    dataset = load_data(conf.data, split='all' if conf.data.name.lower() == 'ffhq' else 'train')
    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.bspp, sampler=datasampler, drop_last=False, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(dataset)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * get_world_size()}')

    # LOAD PRETRAINED VQMODEL
    with main_process_first():
        vqmodel = make_vqmodel(conf.vqmodel.model_name)
    vqmodel = vqmodel.requires_grad_(False).eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained vqmodel: {conf.vqmodel.model_name}')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    wait_for_everyone()

    # START CACHING VQMODEL LATENTS
    logger.info('Start caching vqmodel latents')
    logger.info(f'Cached latents will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    cnt = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Caching', disable=not is_main_process()):
            # get data
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x, y = batch.to(device), None
            B = x.shape[0]
            N = conf.data.img_size // vqmodel.downsample_factor

            # encode image
            enc = vqmodel.encode(x)
            h, quant, idx = enc['h'], enc['quant'], enc['indices'].reshape(B, N * N)

            # encode flipped image
            enc_flip = vqmodel.encode(x.flip(dims=[3]))
            h_flip, quant_flip, idx_flip = enc_flip['h'], enc_flip['quant'], enc_flip['indices'].reshape(B, N * N)

            # gather all processes
            h = torch.cat(gather_tensor(h), dim=0)
            quant = torch.cat(gather_tensor(quant), dim=0)
            idx = torch.cat(gather_tensor(idx), dim=0)
            h_flip = torch.cat(gather_tensor(h_flip), dim=0)
            quant_flip = torch.cat(gather_tensor(quant_flip), dim=0)
            idx_flip = torch.cat(gather_tensor(idx_flip), dim=0)
            if y is not None:
                y = torch.cat(gather_tensor(y), dim=0)

            # save to npz on main process
            if is_main_process():
                for i in range(len(h)):
                    save_path = os.path.join(args.save_dir, f'{cnt}.npz')
                    data = dict(
                        idx=idx[i].cpu().numpy(),
                        idx_flip=idx_flip[i].cpu().numpy(),
                    )
                    if args.full:
                        data.update(
                            h=h[i].cpu().numpy(),
                            quant=quant[i].cpu().numpy(),
                            h_flip=h_flip[i].cpu().numpy(),
                            quant_flip=quant_flip[i].cpu().numpy(),
                        )
                    if y is not None:
                        data.update(y=y[i].cpu().numpy())
                    np.savez(save_path, **data)
                    cnt += 1
    logger.info(f'Cached latents are saved to {args.save_dir}')
    logger.info('Finished caching vqmodel latents')
    cleanup()


if __name__ == '__main__':
    main()
