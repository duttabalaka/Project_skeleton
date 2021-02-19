import skimage
from skimage.color import rgb2gray
from skimage.filters import threshold_triangle, unsharp_mask
import skimage.io
import skimage.measure
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis
import cv2

image = skimage.io.imread(fname='H02.jpg')
image = rgb2gray(image)
print(image.shape)
im = skimage.filters.gaussian(image, sigma=1)
blobs = im < .9 * im.mean()
all_labels = skimage.measure.label(blobs)
print(all_labels)
skel, distance = medial_axis(blobs, return_distance=True)
dist_on_skel = distance * skel
# blobs_labels = skimage.measure.label(blobs)
cv2.imshow("origin", image)
cv2.waitKey(0)
cv2.imshow("result", dist_on_skel)
cv2.waitKey(0)


