import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_triangle
import skimage.io
import skimage.measure
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis
import torch.nn.functional as F


# Filter
def circle(radius):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel


# v = circle(1000)
# plt.imshow(v)
# plt.show()
# print(v)

box = np.zeros((20, 20))
circle_no = 6
for i in range(circle_no):
    cir = circle(i)
    height_pad = (21 - cir.shape[0]) / 2
    width_pad = (21 - cir.shape[1]) / 2
    padded_height = int(np.floor(height_pad))
    padded_height_uneven = int(np.floor(height_pad - 0.5))
    padded_width = int(np.floor(width_pad))
    padded_width_uneven = int(np.floor(width_pad - 0.5))
    final = np.pad(cir, ((padded_height, padded_height_uneven), (padded_width, padded_width_uneven)),
                   mode="constant")
    # final = np.pad(cir,pad_width, 'constant')
    # print(final.shape)
    # plt.imshow(final)
    # plt.show()
    if i == 0:
        stack_arr = final
    else:
        stack_arr = np.dstack((stack_arr, final))
    # print(stack_arr.shape)

filter_data = np.swapaxes(stack_arr, 0, 2)
filter_data = torch.tensor(filter_data)
# print(filter_data.shape)
# 3d to 4d
filter_data = filter_data.unsqueeze(1)
print(filter_data.shape)

# Image Data
image = skimage.io.imread(fname='H02.jpg')
image = rgb2gray(image)
im = skimage.filters.gaussian(image, sigma=1)
blobs = im < .9 * im.mean()
all_labels = skimage.measure.label(blobs)
skel, distance = medial_axis(blobs, return_distance=True)
dist_on_skel = distance * skel
image_data = torch.tensor(dist_on_skel)
# print(image_data.shape)
# 2d to 4d
image_data = image_data.unsqueeze(0)
image_data = image_data.unsqueeze(0)
print(image_data.shape)

final = F.conv2d(image_data, filter_data)
print(final.shape)
plt.imshow(final[0, 3, :], cmap="gray")
plt.show()

# dice r iou performance matrix
# def dice_loss(y_true, y_pred):
#     numerator = 2 * K.sum(y_true * y_pred)
#     denominator = K.sum(y_true) + K.sum(y_pred)
#
#     return 1 - (numerator + 1) / (denominator + 1)