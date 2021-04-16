import numpy as np


def dice_coeff(image_real, final):
    """Computes the dice score of two images

    :param:image_real
    :param: final
    :return: dice score
    """

    numerator = 2 * np.sum(image_real * final)
    denominator = np.sum(image_real) + np.sum(final)
    return numerator / denominator
