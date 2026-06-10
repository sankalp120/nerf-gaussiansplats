import torch

class PositionalEncoding:

    def __init__(self, L=10):
        self.L = L

    def __call__(self, x):
        """
        x: (...,3)

        returns:
        (...,63)
        """

        out = [x]

        for i in range(self.L):

            freq = 2.0 ** i

            out.append(
                torch.sin(freq * x)
            )

            out.append(
                torch.cos(freq * x)
            )

        return torch.cat(out, dim=-1)