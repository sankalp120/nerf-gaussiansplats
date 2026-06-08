import torch
from sampler import sample_points

rays_o = torch.randn(10,3)
rays_d = torch.randn(10,3)

points, t_vals = sample_points(
    rays_o,
    rays_d
)

print(points.shape)
print(t_vals.shape)