from skimage.morphology import medial_axis, dilation
from skimage.filters import threshold_otsu
import cv2


def binarize(img):
    """Returns a binary image fro a grayscale one.

    :param img:
    :return:
    """

    img = cv2.GaussianBlur(img, (1, 1), 0)
    thresh = threshold_otsu(img)  # apply threshold
    blobs = img < thresh
    return blobs



def preprocess(image_input):
    """Returns the radial skeleton

    :param image_input: a graylevel inmage (numpy.array)
    :return: a tensor with the normalised radial skeleton
    """

    blobs = binarize(image_input)
    dilated = dilation(blobs)
    skel, distance = medial_axis(dilated, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel

