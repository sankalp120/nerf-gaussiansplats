import torch
import torch.nn as nn

class TinyNeRF(nn.Module):

    def __init__(self, input_dim=63):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.sigma_head = nn.Linear(256, 1)
        self.rgb_head = nn.Linear(256, 3)

    def forward(self, x):

        h = self.layers(x)

        sigma = torch.relu(
            self.sigma_head(h)
        )

        rgb = torch.sigmoid(
            self.rgb_head(h)
        )

        return rgb, sigma