import torch
import torch.nn.functional as F


def is_torch2_available():
    """Check is torch > 2.0.0 available in env"""

    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):
    """Get generator for image generation"""

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
