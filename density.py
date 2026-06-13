import torch
import torch.nn as nn

class TinyNeRF(nn.Module):

    def __init__(self, input_dim=63):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),

            nn.Linear(256,256),
            nn.ReLU(),

            nn.Linear(256,256),
            nn.ReLU(),

            nn.Linear(256,256),
            nn.ReLU()
        )

        self.rgb_head = nn.Linear(256,3)
        self.sigma_head = nn.Linear(256,1)

    def forward(self,x):

        h = self.mlp(x)

        rgb = torch.sigmoid(
            self.rgb_head(h)
        )

        sigma = torch.relu(
            self.sigma_head(h)
        )

        return rgb,sigma