# import cv2
import numpy as np


def split_img(img: np.array, size_x: int = 128, size_y: int = 128) -> [np.array]:
    """Split image to parts (little images).
    Walk through the whole image by the window of size size_x * size_y without overlays and
    save all parts in list. Images sizes should divide window sizes.
    """
    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += size_x
        curr_y += size_y
    return parts


def normalize_in(img: np.array) -> np.array:
    """"""
    img = img.astype(np.float32)
    img /= 256.0
    img -= 0.5
    return img


def add_border(img: np.array, size_x: int = 128, size_y: int = 128) -> (np.array, int, int):
    """Add border to image, so it will divide window sizes: size_x and size_y"""
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = (size_y - (max_y % size_y) + 1) // 2
        img = cv2.copyMakeBorder(img, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    border_x = 0
    if max_x % size_x != 0:
        border_x = (size_x - (max_x % size_x) + 1) // 2
        img = cv2.copyMakeBorder(img, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img, border_y, border_x


def combine_imgs(imgs: [np.array], max_y: int, max_x: int) -> np.array:
    """Combine image parts to one big image.
    Walk through list of images and create from them one big image with sizes max_x * max_y.
    If border_x and border_y are non-zero, they will be removed from created image.
    The list of images should contain data in the following order:
    from left to right, from top to bottom.
    """
    img = np.zeros((max_y, max_x), np.float)
    size_y, size_x = imgs[0].shape
    curr_y = 0
    i = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            try:
                img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imgs[i]
            except:
                i -= 1
            i += 1
            curr_x += size_x
        curr_y += size_y
    return img


def process_unet_img(img: np.array) -> np.array:
    """Split image to 128x128 parts and run U-net for every part."""
    img, border_y, border_x = add_border(img)
    img = normalize_in(img)
    parts = split_img(img)
    parts = np.array(parts)
    parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2], 1)
    # parts = model.predict(parts, batchsize)
    tmp = []
    for part in parts:
        part.shape = (128, 128)
        tmp.append(part)
    parts = tmp
    img = combine_imgs(parts, img.shape[0], img.shape[1])
    img = img[border_y:img.shape[0] - border_y, border_x:img.shape[1] - border_x]
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def postprocess_img(img: np.array) -> np.array:
    """Apply Otsu threshold to image."""
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def binarize_img(img: np.array) -> np.array:
    """Binarize image, using U-net, Otsu, bottom-hat transform etc."""
    img = process_unet_img(img)
    img = postprocess_img(img)
    return img

# image = cv2.imread('/Users/balakadutta/PycharmProjects/Project_skeleton/H02.jpg', cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image = binarize_img(image)
# cv2.imshow("binary", image)
# cv2.waitKey(0)