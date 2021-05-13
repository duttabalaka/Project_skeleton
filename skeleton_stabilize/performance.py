import numpy as np


def dice_coeff(image_real, final):
    """Computes the dice score of two images

    :param:image_real(torch -> tensor)
    :param: final(torch -> tensor)
    :return: dice score
    """
    image_real, final = image_real.cpu().numpy(), final.cpu().numpy()
    numerator = 2 * np.sum(image_real * final)
    denominator = np.sum(image_real) + np.sum(final)
    return numerator / denominator
