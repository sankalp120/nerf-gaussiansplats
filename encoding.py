import torch

class PositionalEncoding:
    def __init__(self, L=10):
        self.L = L

    def __call__(self, x):
        out = [x]

        for i in range(self.L):
            out.append(torch.sin((2.0 ** i) * x))
            out.append(torch.cos((2.0 ** i) * x))

        return torch.cat(out, dim=-1)