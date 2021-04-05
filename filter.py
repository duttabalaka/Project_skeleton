import torch
import numpy as np
import torch.nn.functional as F


# Filter
def render_circle(radius):
    # todo(balaka) antiliased circle
    kernel = np.zeros((2 * radius, 2 * radius))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel


def create_circular_filter(no_circle, min_radius, max_radius):
    """
    :param no_circle:
    :param min_radius:
    :param max_radius:
    :return: [1 x radius x height x width]
    """
    filter_sz = int(np.ceil(max_radius)) * 2 + 1
    X = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=2)
    X = X - X[:, :, filter_sz//2: filter_sz//2 + 1]

    Y = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=1)
    Y = Y - Y[:, filter_sz//2: filter_sz//2 + 1, :]
    radii = np.linspace(min_radius, max_radius, no_circle).reshape([-1, 1, 1])
    antialiased = - np.clip(((X ** 2 + Y ** 2) ** .5 - radii), -1, 0)

    #in_radi = int((max_radius - min_radius) / (no_circle - 1))
    #c = 0
    # for n in range(len(radii)):
    #     #c = c + in_radi
    #     radius = radii[n]
    #     unpadded_circle = render_circle(radius)
    #     vertical_pad = (((max_radius * 2) + 1) - unpadded_circle.shape[0]) / 2
    #     horizontal_pad = (((max_radius * 2) + 1) - unpadded_circle.shape[1]) / 2
    #     padded_height = int(np.floor(vertical_pad))
    #     padded_height_uneven = int(np.floor(vertical_pad - 0.5))
    #     padded_width = int(np.floor(horizontal_pad))
    #     padded_width_uneven = int(np.floor(horizontal_pad - 0.5))
    #     final = np.pad(unpadded_circle, ((padded_height, padded_height_uneven), (padded_width, padded_width_uneven)),
    #                    mode="constant")
    #     if n == 0:
    #         stack_arr = final
    #     else:
    #         stack_arr = np.dstack((stack_arr, final))
    return torch.tensor(antialiased).unsqueeze(1)
    #filter_data = np.swapaxes(stack_arr, 0, 2)
    #filter_data = np.float32(filter_data)
    #filter_data = torch.tensor(filter_data)
    # 3d to 4d
    #filter_data = filter_data.unsqueeze(1)
    #return filter_data


def circular_filter(image, filter, max_radius):
    final = F.conv2d(image, filter, padding=filter.size(2)//2) # assuming filter width == filter height
    return final

def render_radial_skeleton(radii_stack, radial_skeleton, filter):
    radii_stack, radial_skeleton = torch.tensor(radii_stack), torch.tensor(radial_skeleton)
    in_img = torch.zeros_like(radii_stack)
    for n in range(radii_stack.shape[1]):
        in_img[0,n,:,:] = (radial_skeleton==n)
    stack = F.conv2d(in_img, filter, padding=filter.size(2) // 2, groups = radii_stack.shape[1])  # assuming filter width == filter height
    return stack.sum(dim=1)[0,:,:].numpy() > 0

