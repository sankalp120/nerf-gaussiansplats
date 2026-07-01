import torch

def sample_points(
    rays_o,
    rays_d,
    near=2.0,
    far=6.0,
    N_samples=64,
    perturb=True
):

    t_vals = torch.linspace(
        near,
        far,
        N_samples,
        device=rays_o.device
    )

    if perturb:

        mids = 0.5 * (t_vals[:-1] + t_vals[1:])

        upper = torch.cat(
            [mids, t_vals[-1:]],
            dim=0
        )

        lower = torch.cat(
            [t_vals[:1], mids],
            dim=0
        )

        t_rand = torch.rand(
            (rays_o.shape[0], N_samples),
            device=rays_o.device
        )

        t_vals = (
            lower[None]
            +
            (upper - lower)[None] * t_rand
        )

    else:

        t_vals = t_vals.expand(
            rays_o.shape[0],
            N_samples
        )

    points = (
        rays_o[:, None]
        +
        rays_d[:, None]
        * t_vals[..., None]
    )

    return points, t_vals