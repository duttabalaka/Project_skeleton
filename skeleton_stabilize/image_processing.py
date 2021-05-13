from skimage.morphology import medial_axis, dilation
from skimage.filters import threshold_otsu
import cv2
import torch
import numpy as np


def binarize(img, method="otsu"):
    """Returns a binary image fro a grayscale one.

    :param img:
    :return:
    """
    if method == "otsu":
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
    gray_level = gray_level.cpu().numpy()
    blobs = binarize(gray_level, method="otsu")
    dilated = dilation(blobs)
    skel, distance = medial_axis(dilated, return_distance=True)
    dist_on_skel = distance * skel
    return torch.tensor(dist_on_skel)


def intimg_to_onehot(image_input: torch.Tensor) -> torch.Tensor:
    """

    :param image_input: [BxCxHxW] C==1
    :return: [BxCxHxW] float
    """
    # image_input = image_input.cpu().numpy()
    # no_classes = 2
    # one_hot = np.zeros((no_classes, image_input.shape[0], image_input.shape[1]))
    # for i, unique_value in enumerate(np.unique(image_input)):
    #     one_hot[i, :, :][image_input == unique_value] = 1

    no_classes = 2
    one_hot = torch.nn.functional.one_hot(image_input, no_classes).transpose(1, 4).squeeze(-1)

    return one_hot

