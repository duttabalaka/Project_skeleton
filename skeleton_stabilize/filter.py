import torch
import numpy as np
import torch.nn.functional as F


# Filter


def create_circular_filter(no_circle, min_radius, max_radius, device):
    """
    :param: no_circle
    :param: min_radius
    :param: max_radius
    :return: [1 x radius x height x width]
    """
    filter_sz = int(np.ceil(max_radius)) * 2 + 1
    X = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=2)
    X = X - X[:, :, filter_sz // 2: filter_sz // 2 + 1]
    Y = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=1)
    Y = Y - Y[:, filter_sz // 2: filter_sz // 2 + 1, :]
    radii = np.linspace(min_radius, max_radius, no_circle).reshape([-1, 1, 1])
    antialiased = - np.clip(((X ** 2 + Y ** 2) ** .5 - radii), -1, 0)
    return torch.tensor(antialiased).unsqueeze(1).to(device)


def apply_circular_filter(image:torch.Tensor, filter:torch.Tensor) -> torch.Tensor:
    """
        :param: image
        :param: filter
        :return: image convolve with filter
    """
    with torch.no_grad():
        final = F.conv2d(image, filter, padding=filter.size(2) // 2)  # assuming filter width == filter height
        final = torch.clamp(final, 0., 1.)
    return final


def render_radial_skeleton(radii_stack, radial_skeleton, filter, device):
    """
            :param: radii_stack
            :param: radial_skeleton
            :param: filter
            :return: sum of convolve image
        """
    radii_stack, radial_skeleton = torch.tensor(radii_stack, device=device), torch.tensor(radial_skeleton, device=device)
    in_img = torch.zeros_like(radii_stack)
    for n in range(radii_stack.shape[1]):
        in_img[0, n, :, :] = (radial_skeleton == n)
    stack = F.conv2d(in_img, filter, padding=filter.size(2) // 2,
                     groups=radii_stack.shape[1])  # assuming filter width == filter height
    return (stack.sum(dim=1)[0, :, :] > 0).float()
