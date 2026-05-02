import argparse
import os
import random
import tqdm
import numpy as np
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, required=True, help='Path to directory saving generated samples')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args = get_parser().parse_args()

    # ASSERTIONS
    assert os.path.exists(args.sample_dir)
    assert not os.path.exists(f'{args.sample_dir}.npz')

    # FIND IMAGES RECURSIVELY
    image_paths = []
    for root, _, files in os.walk(args.sample_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    image_paths = sorted(image_paths)
    print(f'Found {len(image_paths)} images in {args.sample_dir}')

    # SHUFFLE IMAGES
    random.seed(args.seed)
    random.shuffle(image_paths)

    # READ IMAGES
    images = []
    for path in tqdm.tqdm(image_paths, desc='Reading images'):
        images.append(np.asarray(Image.open(path)).astype(np.uint8))
    images = np.stack(images)

    # SAVE .NPZ FILE
    np.savez(f'{args.sample_dir}.npz', arr_0=images)
    print(f'Saved .npz file to {args.sample_dir}.npz [shape={images.shape}].')


if __name__ == '__main__':
    main()
