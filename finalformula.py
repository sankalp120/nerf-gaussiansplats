import torch

def volume_render(rgb, sigma, t_vals):

    delta = t_vals[1:] - t_vals[:-1]

    delta = torch.cat([
        delta,
        torch.tensor([1e10], device=t_vals.device)
    ])

    alpha = 1.0 - torch.exp(
        -sigma * delta
    )

    transmittance = torch.cumprod(
        torch.cat([
            torch.ones(
                (alpha.shape[0],1),
                device=alpha.device
            ),
            1.0 - alpha + 1e-10
        ], dim=-1),
        dim=-1
    )[:, :-1]

    weights = alpha * transmittance

    rgb_map = torch.sum(
        weights[...,None] * rgb,
        dim=1
    )

    return rgb_map, weights