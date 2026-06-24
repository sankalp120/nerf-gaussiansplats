# inference.py

import numpy as np
import torch
import imageio.v2 as imageio

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

N_SAMPLES = 64


def render_image(
    pose,
    H,
    W,
    focal,
    model,
    encoder
):

    rays_o, rays_d = get_rays(
        H,
        W,
        focal,
        pose
    )

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rendered_chunks = []

    chunk_size = 1024

    with torch.no_grad():

        for i in range(
            0,
            rays_o.shape[0],
            chunk_size
        ):

            rays_o_chunk = rays_o[
                i:i+chunk_size
            ]

            rays_d_chunk = rays_d[
                i:i+chunk_size
            ]

            points, t_vals = sample_points(
                rays_o_chunk,
                rays_d_chunk,
                N_samples=N_SAMPLES
            )

            encoded = encoder(
                points.reshape(-1,3)
            )

            rgb, sigma = model(
                encoded
            )

            rgb = rgb.reshape(
                rays_o_chunk.shape[0],
                N_SAMPLES,
                3
            )

            sigma = sigma.reshape(
                rays_o_chunk.shape[0],
                N_SAMPLES
            )

            rgb_map, _ = volume_render(
                rgb,
                sigma,
                t_vals
            )

            rendered_chunks.append(
                rgb_map.cpu()
            )

    image = torch.cat(
        rendered_chunks,
        dim=0
    )

    image = image.reshape(
        H,
        W,
        3
    )

    return image.numpy()


def main():

    dataset = NeRFDataset(
        "data/lego/transforms_test.json"
    )

    model = TinyNeRF().to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/nerf_final.pt",
            map_location=DEVICE
        )
    )

    model.eval()

    encoder = PositionalEncoding(
        L=10
    )

    image_gt, pose = dataset[0]

    H = image_gt.shape[0]
    W = image_gt.shape[1]

    focal = (
        0.5 * W
        /
        np.tan(
            0.5 * dataset.camera_angle_x
        )
    )

    pose = pose.to(DEVICE)

    print("Rendering...")

    rendered = render_image(
        pose,
        H,
        W,
        focal,
        model,
        encoder
    )

    rendered = np.clip(
        rendered,
        0,
        1
    )

    imageio.imwrite(
        "rendered.png",
        (rendered * 255).astype(np.uint8)
    )

    print("Saved rendered.png")


if __name__ == "__main__":
    main()