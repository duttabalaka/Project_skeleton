import cv2
import numpy as np


def dice_coeff(image_real, final):
    """Computes the dice score of two images

    :param image_real:
    :param final:
    :return:
    """
    print(type(image_real),type(final))
    #image_real = cv2.normalize(image_real, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #final = 1 - cv2.normalize(final, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    numerator = 2 * np.sum(image_real * final)
    denominator = np.sum(image_real) + np.sum(final)
    return numerator / denominator
