import os
import math
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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

N_RAYS = 1024
N_SAMPLES = 64
LR = 5e-4
NUM_STEPS = 5000

CHECKPOINT_DIR = "checkpoints"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


def train_step(
    image,
    pose,
    focal,
    model,
    encoder,
    optimizer
):

    H, W = image.shape[:2]

    rays_o, rays_d = get_rays(
        H,
        W,
        focal,
        pose
    )

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    image = image.reshape(-1, 3)

    idx = torch.randint(
        0,
        H * W,
        (N_RAYS,),
        device=DEVICE
    )

    rays_o_batch = rays_o[idx]
    rays_d_batch = rays_d[idx]

    target_rgb = image[idx]

    points, t_vals = sample_points(
        rays_o_batch,
        rays_d_batch,
        near=2.0,
        far=6.0,
        N_samples=N_SAMPLES
    )

    points_flat = points.reshape(-1, 3)

    encoded = encoder(
        points_flat
    )

    rgb, sigma = model(
        encoded
    )

    rgb = rgb.reshape(
        N_RAYS,
        N_SAMPLES,
        3
    )

    sigma = sigma.reshape(
        N_RAYS,
        N_SAMPLES
    )

    rgb_pred, _ = volume_render(
        rgb,
        sigma,
        t_vals
    )

    loss = torch.mean(
        (rgb_pred - target_rgb) ** 2
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


def main():

    dataset = NeRFDataset(
        "data/lego/transforms_train.json"
    )

    model = TinyNeRF().to(
        DEVICE
    )

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

    print("\nTraining Started\n")

    for step in range(NUM_STEPS):

        img_idx = random.randint(
            0,
            len(dataset) - 1
        )

        image, pose = dataset[img_idx]

        image = image.to(
            DEVICE
        )

        pose = pose.to(
            DEVICE
        )

        loss = train_step(
            image,
            pose,
            focal,
            model,
            encoder,
            optimizer
        )

        if step % 100 == 0:

            psnr = (
                -10.0
                * math.log10(
                    max(loss, 1e-10)
                )
            )

            print(
                f"Step {step:05d} | "
                f"Loss {loss:.6f} | "
                f"PSNR {psnr:.2f}"
            )

        if step % 1000 == 0 and step > 0:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"nerf_{step}.pt"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Saved {checkpoint_path}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "nerf_final.pt"
        )
    )

    print("\nTraining Complete")


if __name__ == "__main__":
    main()