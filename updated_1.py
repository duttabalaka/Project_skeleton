import cv2
import skimage
import torch
from skimage.color import rgb2gray, gray2rgb
from skimage.morphology import medial_axis
from skimage.filters import threshold_triangle, threshold_otsu
import skimage.io
import numpy as np
from matplotlib import pyplot as plt



# def distance_transform(image):
#     image = skimage.filters.gaussian(image, sigma=1)
#     image = image < threshold_otsu(image)
#     all_labels = skimage.measure.label(image)
#     skel, distance = medial_axis(all_labels, return_distance=True)
#     dist_on_skel = distance * skel
#     return dist_on_skel
#
#
# filepath = cv2.imread('test.png')
# image = rgb2gray(filepath)
# print(image.shape)
# f_img = distance_transform(image)
# print(f_img.shape)
# img = gray2rgb(f_img)
# print(img.shape)
# cv2.imshow("origin", image)
# cv2.waitKey(0)
# cv2.imshow("result", img)
# cv2.waitKey(0)

no_circle=100
min_radius=0
max_radius=99


filter_sz = int(np.ceil(max_radius)) * 2 + 1
X = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=2)
X = X - X[:, :, filter_sz // 2: filter_sz // 2 + 1]
Y = np.cumsum(np.ones([1, filter_sz, filter_sz]), axis=1)
Y = Y - Y[:, filter_sz // 2: filter_sz // 2 + 1, :]
radii = np.linspace(min_radius, max_radius, no_circle).reshape([-1, 1, 1])
antialiased = - np.clip(((X ** 2 + Y ** 2) ** .5 - radii), -1, 0)
print(antialiased.shape)
plt.imshow(antialiased[91,:,:], cmap='gray')
plt.show()
