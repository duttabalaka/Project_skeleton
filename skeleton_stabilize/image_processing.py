from skimage.morphology import medial_axis, dilation
from skimage.filters import threshold_otsu
import cv2


def binarize(img, method="otsu"):
    """Returns a binary image fro a grayscale one.

    :param img:
    :return:
    """
    if method=="otsu":
        return _binarize_otsu(img)
    else:
        raise NotImplemented()


def _binarize_otsu(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    thresh = threshold_otsu(img)  # apply threshold
    blobs = img < thresh
    return blobs


def extract_radial_skeleton(gray_level):
    """Returns the radial skeleton

    :param image_input: a graylevel inmage (numpy.array)
    :return: a tensor with the normalised radial skeleton
    """

    blobs = binarize(gray_level)
    #dilated = dilation(blobs)
    skel, distance = medial_axis(dilated, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel


def intimg_to_onehot(image_input:torch.Tensor)->torch.Tensor:
    """

    :param image_input: [BxCxHxW] C==1
    :return: [BxCxHxW] float
    """
    raise NotImplemented
