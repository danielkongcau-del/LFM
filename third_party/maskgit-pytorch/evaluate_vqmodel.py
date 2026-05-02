import argparse
import os
import tqdm
from omegaconf import OmegaConf

import torch
import torch_fidelity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from metrics import PSNR, SSIM, LPIPS
from models import make_vqmodel
from utils.data import load_data
from utils.logger import get_logger
from utils.image import image_norm_to_float
from utils.misc import set_seed
from utils.experiment import discard_label
from utils.distributed import init_distributed_mode, is_main_process, is_dist_avail_and_initialized
from utils.distributed import wait_for_everyone, cleanup, gather_tensor, get_rank, get_world_size, main_process_first


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to imagenet dataset')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--bspp', type=int, default=64, help='Batch size on each process')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to directory saving samples (for rFID)')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    return parser


def main():
    # PARSE ARGS
    args = get_parser().parse_args()

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
    conf_data = OmegaConf.create(dict(name='imagenet', root=args.dataroot, img_size=args.img_size))
    dataset = load_data(conf_data, split='valid')
    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp, sampler=datasampler, drop_last=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of dataset: {len(dataset)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * get_world_size()}')

    # BUILD MODEL
    with main_process_first():
        vqmodel = make_vqmodel(args.model_name)
    vqmodel.eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load {args.model_name} vqmodel')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    logger.info('=' * 50)
    wait_for_everyone()

    # START EVALUATION
    logger.info('Start evaluating...')
    idx = 0
    if args.save_dir is not None:
        os.makedirs(os.path.join(args.save_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'reconstructed'), exist_ok=True)

    psnr_fn = PSNR(reduction='none')
    ssim_fn = SSIM(reduction='none')
    lpips_fn = LPIPS(reduction='none').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for x in tqdm.tqdm(dataloader, desc='Evaluating', disable=not is_main_process()):
            x = discard_label(x).to(device)
            recx = vqmodel(x)
            recx = recx.clamp(-1, 1)

            x = image_norm_to_float(x)
            recx = image_norm_to_float(recx)
            psnr = psnr_fn(recx, x)
            ssim = ssim_fn(recx, x)
            lpips = lpips_fn(recx, x)

            psnr = torch.cat(gather_tensor(psnr), dim=0)
            ssim = torch.cat(gather_tensor(ssim), dim=0)
            lpips = torch.cat(gather_tensor(lpips), dim=0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips)

            if args.save_dir is not None:
                x = torch.cat(gather_tensor(x), dim=0)
                recx = torch.cat(gather_tensor(recx), dim=0)
                if is_main_process():
                    for ori, dec in zip(x, recx):
                        save_image(ori, os.path.join(args.save_dir, 'original', f'{idx}.png'))
                        save_image(dec, os.path.join(args.save_dir, 'reconstructed', f'{idx}.png'))
                        idx += 1

    psnr = torch.cat(psnr_list, dim=0).mean().item()
    ssim = torch.cat(ssim_list, dim=0).mean().item()
    lpips = torch.cat(lpips_list, dim=0).mean().item()

    logger.info(f'PSNR: {psnr:.4f}')
    logger.info(f'SSIM: {ssim:.4f}')
    logger.info(f'LPIPS: {lpips:.4f}')

    if is_main_process() and args.save_dir is not None:
        fid_score = torch_fidelity.calculate_metrics(
            input1=os.path.join(args.save_dir, 'original'),
            input2=os.path.join(args.save_dir, 'reconstructed'),
            fid=True, verbose=False,
        )['frechet_inception_distance']
        logger.info(f'rFID: {fid_score:.4f}')

    cleanup()


if __name__ == '__main__':
    main()
