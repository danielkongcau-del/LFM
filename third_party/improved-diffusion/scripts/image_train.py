"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    sample_diffusion = None
    if args.sample_interval > 0:
        logger.log("creating sample diffusion...")
        sample_diffusion = create_gaussian_diffusion(
            steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            sigma_small=args.sigma_small,
            noise_schedule=args.noise_schedule,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            timestep_respacing=args.sample_timestep_respacing,
        )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        keep_latest=args.keep_latest,
        sample_diffusion=sample_diffusion,
        sample_interval=args.sample_interval,
        sample_num_samples=args.sample_num_samples,
        sample_batch_size=args.sample_batch_size,
        sample_image_size=args.image_size,
        sample_use_ddim=args.sample_use_ddim,
        sample_clip_denoised=args.sample_clip_denoised,
        sample_dir=args.sample_dir,
        sample_keep_latest=args.sample_keep_latest,
        sample_save_raw=args.sample_save_raw,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        keep_latest=False,
        sample_interval=0,
        sample_num_samples=16,
        sample_batch_size=8,
        sample_timestep_respacing="ddim20",
        sample_use_ddim=True,
        sample_clip_denoised=True,
        sample_dir="",
        sample_keep_latest=False,
        sample_save_raw=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
