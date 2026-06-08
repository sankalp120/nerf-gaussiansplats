import torch
from rays import get_rays

H = 100
W = 100
focal = 100

c2w = torch.eye(4)

rays_o, rays_d = get_rays(H, W, focal, c2w)

print(rays_o.shape)
print(rays_d.shape)