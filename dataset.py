# dataset.py

import os
import json
import imageio.v2 as imageio
import numpy as np
import torch


class NeRFDataset:

    def __init__(self, json_path):

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]

        self.images = []
        self.poses = []

        base_dir = os.path.dirname(json_path)

        print(f"Loading dataset from: {base_dir}")

        for frame in self.frames:

            image_path = os.path.join(
                base_dir,
                frame["file_path"] + ".png"
            )

            image_path = os.path.normpath(image_path)

            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image not found:\n{image_path}"
                )

            image = imageio.imread(image_path)

            image = image.astype(np.float32) / 255.0

            # RGBA -> RGB
            if image.shape[-1] == 4:
                image = image[..., :3]

            pose = np.array(
                frame["transform_matrix"],
                dtype=np.float32
            )

            self.images.append(image)
            self.poses.append(pose)

        self.images = torch.tensor(
            np.stack(self.images),
            dtype=torch.float32
        )

        self.poses = torch.tensor(
            np.stack(self.poses),
            dtype=torch.float32
        )

        print(
            f"Loaded {len(self.images)} images"
        )

        print(
            f"Image shape: {self.images[0].shape}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        return (
            self.images[idx],
            self.poses[idx]
        )