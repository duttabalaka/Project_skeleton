import torch
import numpy as np
import torch.nn.functional as F


# Filter
def circle(radius):
    kernel = np.zeros((2 * radius, 2 * radius))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel


def create_circular_filter(no_circle, min_radius, max_radius):
    in_radi = int((max_radius - min_radius) / (no_circle - 1))
    c = 0
    for i in range(no_circle):
        c = c + in_radi
        cir = circle(c)
        height_pad = (((max_radius * 2) + 1) - cir.shape[0]) / 2
        width_pad = (((max_radius * 2) + 1) - cir.shape[1]) / 2
        padded_height = int(np.floor(height_pad))
        padded_height_uneven = int(np.floor(height_pad - 0.5))
        padded_width = int(np.floor(width_pad))
        padded_width_uneven = int(np.floor(width_pad - 0.5))
        final = np.pad(cir, ((padded_height, padded_height_uneven), (padded_width, padded_width_uneven)),
                       mode="constant")
        if i == 0:
            stack_arr = final
        else:
            stack_arr = np.dstack((stack_arr, final))
    filter_data = np.swapaxes(stack_arr, 0, 2)
    filter_data = np.float32(filter_data)
    filter_data = torch.tensor(filter_data)
    # 3d to 4d
    filter_data = filter_data.unsqueeze(1)
    return filter_data


def circular_filter(image, filter, max_radius):
    kernel_size = ((max_radius*2), (max_radius*2))
    padding = []
    for k in kernel_size:
        if k % 2 == 0:
            pad = [(k - 1) // 2, (k - 1) // 2 + 1]
        else:
            pad = [(k - 1) // 2, (k - 1) // 2]
        padding.extend(pad)

    x = F.pad(image, pad=padding)
    final = F.conv2d(x, filter)
    return final
