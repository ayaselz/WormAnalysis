import os.path
import numpy as np
import cv2

from parameters import Parameters
from exceptions import OpenImageError


def open_image(path: str, num: int, flip: bool):
    """Helper function to import image data from path. Return 16-bit and
    8-bit images for further operation.

    :param
    """
    if path is None or path == "":
        raise OpenImageError(f"Image path is empty. Image number: {num}")

    image_file = os.path.join(path, f'{num:04}' + '.tif')
    image_16bit, image_8bit = transfer_16bit_to_8bit(image_file)
    # try again to get the image data by using another path formation
    # 再次尝试获取图片的信息（用另一种可能的路径格式）
    if image_16bit is None:
        image_file = os.path.join(path, f'{num}' + '.tif')
        image_16bit, image_8bit = transfer_16bit_to_8bit(image_file)

    if image_16bit is None:
        raise OpenImageError(f"Image content is empty. Image number: {num}")
    # flip the image if `flip` is true
    if flip:
        image_8bit = cv2.flip(image_8bit, 1)
        image_16bit = cv2.flip(image_16bit, 1)
    print(type(image_16bit))
    return image_16bit, image_8bit


def transfer_16bit_to_8bit(image_path: str):
    """Helper function for function open_image()"""
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_16bit is None:
        return None, None
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    image_8bit \
        = np.array(np.rint(
            255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))),
            dtype=np.uint8)
    return image_16bit, image_8bit


class Image(object):
    """
    Image() maintain inform of a single image, including image number and
    path. Further, inside the class image is processed into data form and
    highlights (potential neurons) will be located.
    """

    def __init__(self, image_path: str, image_num: int,
                 parameters: Parameters, flip: bool) -> None:
        # inform from outside
        self.path = image_path
        self.num = image_num
        self.parameters = parameters
        self.flip = flip
        # inside repository
        # --- 16-bit image and 8-bit image ---
        self.bit16, self.bit8 = open_image(self.path, self.num, self.flip)
        # ---  ---

    def potential_neurons(self, image, circle=6, ratio=0.4) -> list:
        """
        Find the brightest points in the image as potential neurons. They will
        be further processed inside Neurons().
        """
        potential_neurons = []
        image_max = np.max(image)
        start = int(image.shape[1] / 2)
        end = int(image.shape[0])
        for column in range(start, end):
            for row in range(0, image.shape[1]):
                if image[row][column] > (ratio * image_max):
                    if self.surrender(image, row, column, circle):
                        potential_neurons.append([row, column])
        return potential_neurons
