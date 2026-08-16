import random
import torch
import torch.nn.functional as F


def gaussian_noise(x, sigma_min=0.01, sigma_max=0.08):
    sigma = random.uniform(sigma_min, sigma_max)
    return x + torch.randn_like(x) * sigma


def speckle_noise(x, strength_min=0.03, strength_max=0.20):
    strength = random.uniform(strength_min, strength_max)
    return x + x * torch.randn_like(x) * strength


def random_downsample_upsample(x, min_scale=2, max_scale=4):
    scale = random.randint(min_scale, max_scale)
    h, w = x.shape[-2:]
    nh, nw = max(8, h // scale), max(8, w // scale)

    y = F.interpolate(x, size=(nh, nw), mode="area")
    return F.interpolate(
        y,
        size=(h, w),
        mode="bicubic",
        align_corners=False
    )


def randomized_degradation(x):
    ops = [
        gaussian_noise,
        speckle_noise,
        random_downsample_upsample
    ]

    random.shuffle(ops)

    for op in ops:
        x = op(x)

    return x
