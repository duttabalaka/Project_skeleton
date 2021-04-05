import torch
from skimage.morphology import medial_axis
import cv2


def preprocess(image_input):
    im = cv2.GaussianBlur(image_input, (5, 5), 0)
    blobs = im < .9 * im.mean()
    skel, distance = medial_axis(blobs, return_distance=True)
    dist_on_skel = distance * skel
    # norm_image = cv2.normalize(dist_on_skel, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # norm_image = cv2.GaussianBlur(norm_image, (5, 5), 0)
    return torch.tensor(dist_on_skel)
