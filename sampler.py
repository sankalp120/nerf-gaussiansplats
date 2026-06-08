import torch

def sample_points(
    rays_o,
    rays_d,
    near=2.0,
    far=6.0,
    N_samples=64
):
    """
    rays_o : (N_rays,3)
    rays_d : (N_rays,3)

    returns:
        points : (N_rays,N_samples,3)
        t_vals : (N_samples,)
    """

    t_vals = torch.linspace(
        near,
        far,
        N_samples,
        device=rays_o.device
    )

    points = (
        rays_o[:, None, :]
        +
        rays_d[:, None, :]
        * t_vals[None, :, None]
    )

    return points, t_vals