import json
import imageio.v2 as imageio
import numpy as np
import torch

class NeRFDataset:

    def __init__(
        self,
        json_path,
        image_scale=1
    ):

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]

        self.frames = meta["frames"]

        self.images = []
        self.poses = []

        for frame in self.frames:

            image_path = (
                frame["file_path"] + ".png"
            )

            image = imageio.imread(
                image_path
            )

            image = image.astype(
                np.float32
            ) / 255.0

            self.images.append(image)

            pose = np.array(
                frame["transform_matrix"],
                dtype=np.float32
            )

            self.poses.append(pose)

        self.images = torch.tensor(
            np.stack(self.images)
        )

        self.poses = torch.tensor(
            np.stack(self.poses)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        return (
            self.images[idx],
            self.poses[idx]
        )