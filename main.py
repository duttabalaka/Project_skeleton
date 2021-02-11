
import cv2 as cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize
import skimage
import numpy as np

#  Method1
# def ske(img):
#     img = img.copy()
#     skel = img.copy()
#
#     skel[:, :] = 0
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#
#     while True:
#         eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
#         temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
#         temp = cv2.subtract(img, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img[:, :] = eroded[:, :]
#         if cv2.countNonZero(img) == 0:
#             break
#
#     return skel

def ske(img):
    # skel = img.copy()
    # skel[:, :] = 0
    skel= ndimage.distance_transform_cdt(img)
    # skel =ndimage.distance_transform_edt(img)
    # skel =ndimage.distance_transform_bf(img)
    # print(skel.shape)
    norm_skel =  skel/skel.max()
    norm_skel[norm_skel< 0.98] = 0
    norm_skel[norm_skel > 0.98] = 1
    kernel = np.ones((2,2),np.uint8)
    #
    # while True:
    eroded = cv2.erode(norm_skel, kernel)
    # temp = cv2.dilate(eroded, kernel)
    # temp = cv2.subtract(skel, temp)
    # skel = cv2.bitwise_or(skel, temp)
    #     img = eroded.copy()

    return eroded


f, ax = plt.subplots(2, 2)
img = np.zeros([200, 200]);
img[20:180, 20:180] = 1;
img[50:150, 50:150] = 0
ax[0][0].imshow(img)
# ax[0][1].imshow(skeletonize(img))
s1_img = ske(img)
ax[0][1].imshow(s1_img)

# adding 4 pixels on the border of the object
img[18:20, 40:42] = 1
ax[1][0].imshow(img)
# ax[1][1].imshow(skeletonize(img))
s_img = ske(img)
ax[1][1].imshow(s_img)
plt.show()

