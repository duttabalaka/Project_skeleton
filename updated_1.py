import cv2
import skimage
from skimage.color import rgb2gray, gray2rgb
from skimage.morphology import medial_axis
from skimage.filters import threshold_triangle, threshold_otsu
import skimage.io


def distance_transform(image):
    image = skimage.filters.gaussian(image, sigma=1)
    image = image < threshold_otsu(image)
    all_labels = skimage.measure.label(image)
    skel, distance = medial_axis(all_labels, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel


filepath = cv2.imread('test.png')
image = rgb2gray(filepath)
print(image.shape)
f_img = distance_transform(image)
print(f_img.shape)
img = gray2rgb(f_img)
print(img.shape)
cv2.imshow("origin", image)
cv2.waitKey(0)
cv2.imshow("result", img)
cv2.waitKey(0)
