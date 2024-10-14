import torch


def generate_noise(noise_type, x, timestep=None, generator=None):
    noise = torch.randn(
        x.shape, dtype=x.dtype, layout=x.layout, generator=generator
    ).to(x.device)
    return noise
