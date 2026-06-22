# train.py

import random
import numpy as np
import torch

from dataset import NeRFDataset
from rays import get_rays
from sampler import sample_points
from encoding import PositionalEncoding
from model import TinyNeRF
from render import volume_render


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_RAYS = 1024
N_SAMPLES = 64
NUM_STEPS = 10000
LR = 5e-4


def train_step(
    image,
    pose,
    focal,
    model,
    encoder,
    optimizer
):

    H, W = image.shape[:2]

    # -------------------------
    # Generate Rays
    # -------------------------

    rays_o, rays_d = get_rays(
        H,
        W,
        focal,
        pose
    )

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    image = image.reshape(-1, 3)

    # -------------------------
    # Random Ray Batch
    # -------------------------

    idx = torch.randint(
        0,
        H * W,
        (N_RAYS,),
        device=DEVICE
    )

    rays_o_batch = rays_o[idx]
    rays_d_batch = rays_d[idx]

    target_rgb = image[idx]

    # -------------------------
    # Sample Points
    # -------------------------

    points, t_vals = sample_points(
        rays_o_batch,
        rays_d_batch,
        near=2.0,
        far=6.0,
        N_samples=N_SAMPLES
    )

    # -------------------------
    # Flatten Points
    # -------------------------

    points_flat = points.reshape(-1, 3)

    # -------------------------
    # Positional Encoding
    # -------------------------

    encoded = encoder(points_flat)

    # -------------------------
    # NeRF Forward
    # -------------------------

    rgb, sigma = model(encoded)

    rgb = rgb.reshape(
        N_RAYS,
        N_SAMPLES,
        3
    )

    sigma = sigma.reshape(
        N_RAYS,
        N_SAMPLES
    )

    # -------------------------
    # Volume Rendering
    # -------------------------

    rgb_pred, _ = volume_render(
        rgb,
        sigma,
        t_vals
    )

    # -------------------------
    # MSE Loss
    # -------------------------

    loss = torch.mean(
        (rgb_pred - target_rgb) ** 2
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


def main():

    dataset = NeRFDataset(
        "transforms_train.json"
    )

    model = TinyNeRF().to(DEVICE)

    encoder = PositionalEncoding(
        L=10
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    sample_img, _ = dataset[0]

    H = sample_img.shape[0]
    W = sample_img.shape[1]

    focal = (
        0.5 * W
        /
        np.tan(
            0.5 * dataset.camera_angle_x
        )
    )

    for step in range(NUM_STEPS):

        img_idx = random.randint(
            0,
            len(dataset) - 1
        )

        image, pose = dataset[img_idx]

        image = image[..., :3]

        image = image.to(
            DEVICE
        ).float()

        pose = pose.to(
            DEVICE
        ).float()

        loss = train_step(
            image,
            pose,
            focal,
            model,
            encoder,
            optimizer
        )

        if step % 100 == 0:

            print(
                f"Step {step:05d} | Loss {loss:.6f}"
            )

        if step % 1000 == 0 and step > 0:

            torch.save(
                model.state_dict(),
                f"checkpoint_{step}.pt"
            )

    torch.save(
        model.state_dict(),
        "nerf_final.pt"
    )

    print("Training complete")


if __name__ == "__main__":
    main()