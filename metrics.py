import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dataset import NeRFDataset

import imageio.v2 as imageio

dataset = NeRFDataset(
    "data/lego/transforms_test.json"
)

gt, _ = dataset[0]

pred = imageio.imread(
    "rendered.png"
).astype(np.float32) / 255.0

psnr = peak_signal_noise_ratio(
    gt,
    pred,
    data_range=1.0
)

ssim = structural_similarity(
    gt,
    pred,
    channel_axis=2,
    data_range=1.0
)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")