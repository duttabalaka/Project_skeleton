import torch
from skimage.morphology import medial_axis
from skimage.filters import threshold_otsu
import cv2

def binarize(img):
    """Returns a binary image fro a grayscale one.

    :param img:
    :return:
    """
    # blobs = im < .9 * im.mean()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = threshold_otsu(img)  # apply threshold
    blobs = img < thresh
    return blobs

def preprocess(image_input, normalize=False):
    """Returns the radial skeleton

    :param image_input: a graylevel inmage (numpy.array)
    :return: a tensor with the normalised radial skeleton
    """
    blobs = binarize(image_input)
    skel, distance = medial_axis(blobs, return_distance=True)
    dist_on_skel = distance * skel
    if normalize:
        return cv2.normalize(dist_on_skel, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        return dist_on_skel