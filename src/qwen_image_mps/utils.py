import torch


def create_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)
