import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_triangle
import skimage.io
import skimage.measure
from matplotlib import pyplot as plt
from skimage.filters import threshold_triangle, unsharp_mask, threshold_otsu
from skimage.morphology import medial_axis
import torch.nn.functional as F
import cv2 as cv2


# Filter
def circle(radius):
    kernel = np.zeros((2 * radius, 2 * radius))
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel


# v = circle(1000)
# plt.imshow(v)
# plt.show()
# print(v)
min_radius = 2
max_radius = 10
no_circle = 5
in_radi = int((max_radius - min_radius)/(no_circle - 1))
c = 0
for i in range(no_circle):
    c = c + in_radi
    cir = circle(c)
    print("circle",cir.shape)
    height_pad = (((max_radius*2)+1) - cir.shape[0]) / 2
    width_pad = (((max_radius*2)+1)- cir.shape[1]) / 2
    padded_height = int(np.floor(height_pad))
    print("pad1", padded_height)
    padded_height_uneven = int(np.floor(height_pad - 0.5))
    print("pad2", padded_height_uneven)
    padded_width = int(np.floor(width_pad))
    padded_width_uneven = int(np.floor(width_pad - 0.5))
    final = np.pad(cir, ((padded_height, padded_height_uneven), (padded_width, padded_width_uneven)),
                   mode="constant")
    if i == 0:
        stack_arr = final
    else:
        stack_arr = np.dstack((stack_arr, final))
    # print(stack_arr.shape)

filter_data = np.swapaxes(stack_arr, 0, 2)
filter_data = np.float32(filter_data)
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
norm_image = cv2.normalize(dist_on_skel, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_image = skimage.filters.gaussian(norm_image, sigma=.1)
image_data = torch.tensor(norm_image)
cv2.imshow("image_data", norm_image)
print(image_data.shape)
# 2d to 4d
image_data = image_data.unsqueeze(0)
image_data = image_data.unsqueeze(0)
print(image_data.shape)

# input compare
image_c = skimage.io.imread(fname='test.png')
image_c = torch.tensor(image_c)
# image_c = rgb2gray(image)

# 2d to 4d
image_data_c = image_c.unsqueeze(0)
image_data_c = image_c.unsqueeze(0)
print("0 =" ,image_data_c.shape)

kernel_size = (max_radius*2, max_radius*2)
padding = []
for k in kernel_size:
    if k % 2 == 0:
        pad = [(k - 1) // 2, (k - 1) // 2 + 1]
    else:
        pad = [(k - 1) // 2, (k - 1) // 2]
    padding.extend(pad)

x = F.pad(image_data, pad=padding)
final = F.conv2d(x, filter_data)
# print(final.shape)
# final = 255 - final
# final[final >= 10] = 255
# final[final <= 10] = 0
final_first = torch.Tensor.numpy(final[0, 2, :])
print("1 =", final_first.shape)
cv2.imshow("final",final_first)


# dice r iou performance matrix
def dice_coeff(image_real, final):
    image_real = torch.Tensor.numpy(image_real)
    print("2 = ", image_real.shape)
    # final = torch.Tensor.numpy(final)
    image_real = cv2.normalize(image_real, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    final = 1 - cv2.normalize(final, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("baal2", final)
    numerator = 2 * np.sum(image_real * final)
    denominator = np.sum(image_real) + np.sum(final)
    return numerator / denominator


loss = dice_coeff(image_data_c, final_first)
print(loss)

cv2.waitKey(0)