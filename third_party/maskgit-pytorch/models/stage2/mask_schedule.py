import numpy as np


def get_mask_scheduling_fn(mask_schedule_type: str):
    if mask_schedule_type == 'linear':
        fn = lambda r: 1 - r
    elif mask_schedule_type == 'cosine':
        fn = lambda r: np.cos(r * np.pi / 2)
    elif mask_schedule_type == 'arccos':
        fn = lambda r: np.arccos(r) / (np.pi / 2)
    elif mask_schedule_type.startswith('pow'):
        exponent = float(mask_schedule_type[3:])
        fn = lambda r: 1 - r ** exponent
    else:
        raise ValueError(f"Unknown mask schedule type: {mask_schedule_type}")

    fn_clip = lambda r: np.clip(fn(r), 0., 1.)
    fn_force = lambda r: np.where(r == 1, 0., fn_clip(r))
    return fn_force
