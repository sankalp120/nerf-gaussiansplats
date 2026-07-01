import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):

    def __init__(self, input_dim=63):

        super().__init__()

        self.fc1 = nn.Linear(input_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)

        self.fc5 = nn.Linear(
            256 + input_dim,
            256
        )

        self.fc6 = nn.Linear(256,256)
        self.fc7 = nn.Linear(256,256)
        self.fc8 = nn.Linear(256,256)

        self.sigma = nn.Linear(
            256,
            1
        )

        self.feature = nn.Linear(
            256,
            256
        )

        self.rgb = nn.Linear(
            256,
            3
        )

    def forward(self,x):

        input_copy = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = torch.cat(
            [x,input_copy],
            dim=-1
        )

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))

        sigma = F.relu(
            self.sigma(x)
        )

        feature = F.relu(
            self.feature(x)
        )

        rgb = torch.sigmoid(
            self.rgb(feature)
        )

        return rgb,sigma