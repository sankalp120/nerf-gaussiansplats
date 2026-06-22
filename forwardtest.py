import numpy as np
import torch

from dataset import NeRFDataset
from rays import get_rays
from sampler import sample_points
from encoding import PositionalEncoding
from model import TinyNeRF
from render import volume_render

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = NeRFDataset(
    "data/lego/transforms_train.json"
)

image, pose = dataset[0]

H, W = image.shape[:2]

focal = (
    0.5 * W
    /
    np.tan(
        0.5 * dataset.camera_angle_x
    )
)

rays_o, rays_d = get_rays(
    H,
    W,
    focal,
    pose
)

# Flatten rays
rays_o = rays_o.reshape(-1,3)
rays_d = rays_d.reshape(-1,3)

# Small batch
rays_o = rays_o[:128]
rays_d = rays_d[:128]

points, t_vals = sample_points(
    rays_o,
    rays_d,
    N_samples=64
)

print("Points:", points.shape)

encoder = PositionalEncoding(L=10)

encoded = encoder(
    points.reshape(-1,3)
)

print("Encoded:", encoded.shape)

model = TinyNeRF()

rgb, sigma = model(encoded)

print("RGB:", rgb.shape)
print("Sigma:", sigma.shape)

rgb = rgb.reshape(
    128,
    64,
    3
)

sigma = sigma.reshape(
    128,
    64
)

rgb_map, weights = volume_render(
    rgb,
    sigma,
    t_vals
)

print("Rendered:", rgb_map.shape)