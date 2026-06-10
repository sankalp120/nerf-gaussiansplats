import torch
from encoding import PositionalEncoding

encoder = PositionalEncoding(L=10)

x = torch.rand(5,3)

encoded = encoder(x)

print(encoded.shape)

x = torch.tensor([[1.0,0.5,0.25]])

encoder = PositionalEncoding(L=3)

print(encoder(x))