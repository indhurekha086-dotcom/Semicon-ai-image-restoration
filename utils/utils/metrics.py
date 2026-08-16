import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def psnr(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def ssim(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if pred.ndim == 3:
        pred = np.moveaxis(pred, 0, -1)
        target = np.moveaxis(target, 0, -1)
        return structural_similarity(
            target,
            pred,
            data_range=1.0,
            channel_axis=-1
        )

    return structural_similarity(
        target,
        pred,
        data_range=1.0
    )
