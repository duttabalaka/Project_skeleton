import cv2 as cv2
import imageio
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.draw import draw
from skimage.exposure import exposure
from skimage.filters import threshold_triangle, unsharp_mask, threshold_otsu
from skimage.morphology import skeletonize, medial_axis, opening, closing, remove_small_objects, dilation
import skimage
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as distmap

# Method:1
# def ske(img):
#     skel = ndimage.distance_transform_cdt(img)
#     # skel =ndimage.distance_transform_edt(img)
#     # skel =ndimage.distance_transform_bf(img)
#
#     norm_skel = skel / skel.max()
#     norm_skel[norm_skel < 0.8] = 0
#     norm_skel[norm_skel > 0.8] = 1
#     kernel = np.ones((1,1),np.uint8)
#     eroded = cv2.erode(norm_skel, kernel)
#
#     return eroded
from skimage.segmentation import clear_border

img = np.zeros([200, 200]);
img = cv2.circle(img, center=(100, 100), radius=50, color=(255, 255, 255), thickness=20)

# with distance transform
# cv2.imshow("img",img)
# cv2.waitKey(0)
# s_img=ske(img)
# cv2.imshow("result",s_img)
# cv2.waitKey(0)

# with medical axis
# skel, distance = medial_axis(img, return_distance=True)
# dist_on_skel = distance * skel
# cv2.imshow("result", dist_on_skel)
# cv2.waitKey(0)


# method 2
# sample_img_path = "/Users/balakadutta/PycharmProjects/Project_skeleton/test.png"
# sample_img = imageio.imread(sample_img_path)
#
# cv2.imshow("img",sample_img)
# cv2.waitKey(0)

# try1 (bad)
# im = cv2.imread('/Users/balakadutta/PycharmProjects/Project_skeleton/test.png', cv2.IMREAD_GRAYSCALE)
# skel, distance = medial_axis(im, return_distance=True)
# # Distance to the background for pixels of the skeleton
# dist_on_skel = distance * skel
# skel_morph = skeletonize(im > 0)
#
# # Distance to the background for pixels of the skeleton
# dist_on_skel = distance * skel
#
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))
# ax1.imshow(distance, cmap = 'gray', interpolation='nearest')
# ax1.contour(im)
# ax1.axis('off')
# ax1.set_title('Distance Map')
# ax2.imshow(skel_morph, cmap = 'gray')
# ax2.set_title('Skeleton')
# ax2.axis('off')
# ax3.imshow(dist_on_skel, cmap = 'gray', interpolation='nearest')
# ax3.contour(im)
# ax3.axis('off')
#
# fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
# plt.show()

# method 3

image = cv2.imread('/Users/balakadutta/PycharmProjects/Project_skeleton/test.png', cv2.IMREAD_GRAYSCALE)


# skel, distance = medial_axis(im, return_distance=True)
def preprocess_img(im, rescaling=True, sharpening=True):
    if rescaling:  # rescale the intensity of the pixel values
        p2, p99 = np.percentile(im, (2, 99))
        im = exposure.rescale_intensity(im, out_range=(p2, p99))

    if sharpening and rescaling:
        im = unsharp_mask(im)

    thresh = threshold_otsu(im)  # apply threshold
    mask = im < thresh
    # mask = remove_small_objects(mask > 0)
    mask = dilation(mask)
    # mask = clear_border(mask)  # remove image border artifacts
    return mask  # remove image border artifacts


img = preprocess_img(image)
img1 = np.uint8(img)
skel, distance = medial_axis(img1, return_distance=True)
dist_on_skel = distance * skel

cv2.imshow("origin", image)
cv2.waitKey(0)
cv2.imshow("result", dist_on_skel)
cv2.waitKey(0)
