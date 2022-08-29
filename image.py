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

    # image_file = os.path.join(path, f'{num:04}' + '.tif')
    image_16bit, image_8bit = transfer_16bit_to_8bit(path)
    # try again to get the image data by using another path formation
    # 再次尝试获取图片的信息（用另一种可能的路径格式） | 直接导入path应该就没这个问题
    # if image_16bit is None:
    #     image_file = os.path.join(path, f'{num}' + '.tif')
    #     image_16bit, image_8bit = transfer_16bit_to_8bit(image_file)

    if image_16bit is None:
        raise OpenImageError(f"Image content is empty. Image number: {num}")
    # flip the image if `flip` is true
    if flip:
        image_8bit = cv2.flip(image_8bit, 1)
        image_16bit = cv2.flip(image_16bit, 1)

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


def surrender(image, row: int, column: int, circle: int) -> bool:
    """
    Helper function for Image.potential_neurons().
    :param image:
    :param row:
    :param column:
    :param circle:
    :return: true iff the point in (row, column) is a highlight comparing to
    pixels around
    """
    for i in range(1, circle + 1):
        if not surrender_value_compare(image, row, column, i):
            return False
    return True


def surrender_value_compare(image, row: int, column: int, radius: int) -> bool:
    """
    Helper function for function surrender(),
    to compare highlight round by round.
    :param image:
    :param row:
    :param column:
    :param radius:
    :return:
    """
    value = image[row][column]
    for i in range(0, 2 * radius + 1):
        if 0 < row - radius + i < image.shape[0] \
                and 0 < column - radius \
                and column + radius < image.shape[1]:
            if value < image[row - radius + i][column - radius] \
                    or value < image[row - radius + i][column + radius]:
                return False
    for i in range(1, 2 * radius):
        if 0 < row - radius \
                and row + radius < image.shape[0] \
                and 0 < column - radius + i < image.shape[1]:
            if value < image[row - radius][column - radius + i] \
                    or value < image[row + radius][column - radius + i]:
                return False
    return True


def draw_rectangle(image, row: int, column: int,
                   label_text: str,
                   radius=5, text_place=5) -> None:
    """
    Helper to draw rectangle and labelled text for the given point (column,row).

    :param image:
    :param column:
    :param row:
    :param label_text:
    :param radius:
    :param text_place:
    :return:
    """
    cv2.rectangle(image, (column - radius, row - radius),
                  (column + radius, row + radius), 255)
    cv2.putText(image, label_text, (column - radius, row - radius - text_place),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)


class ImageInform(object):
    """
    Restore image information for each flame.
    """

    def __init__(self, num: int = None,
                 right_row: int = None, right_column: int = None,
                 right_brightness = None, right_black = None,
                 left_row: int = None, left_column: int = None,
                 left_brightness = None, left_black = None):
        self.__num = num
        self.__right_row = right_row
        self.__right_column = right_column
        self.__right_brightness = right_brightness
        self.__right_black = right_black
        self.__left_row = left_row
        self.__left_column = left_column
        self.__left_brightness = left_brightness
        self.__left_black = left_black
        self.__brightness = left_brightness / right_brightness

    @property
    def num(self) -> int:
        return self.__num

    @num.setter
    def num(self, num: int) -> None:
        self.__num = num

    @property
    def right_row(self) -> int:
        return self.__right_row

    @right_row.setter
    def right_row(self, right_row: int) -> None:
        self.__right_row = right_row

    @property
    def right_column(self) -> int:
        return self.__right_column

    @right_column.setter
    def right_column(self, right_column: int) -> None:
        self.__right_column = right_column

    @property
    def right_brightness(self):
        return self.__right_brightness

    @right_brightness.setter
    def right_brightness(self, right_brightness) -> None:
        self.__right_brightness = right_brightness

    @property
    def right_black(self):
        return self.__right_black

    @right_black.setter
    def right_black(self, right_black) -> None:
        self.__right_black = right_black

    @property
    def left_row(self) -> int:
        return self.__left_row

    @left_row.setter
    def left_row(self, left_row: int) -> None:
        self.__left_row = left_row

    @property
    def left_column(self) -> int:
        return self.__left_column

    @left_column.setter
    def left_column(self, left_column: int) -> None:
        self.__left_column = left_column

    @property
    def left_brightness(self):
        return self.__left_brightness

    @left_brightness.setter
    def left_brightness(self, left_brightness) -> None:
        self.__left_brightness = left_brightness

    @property
    def left_black(self):
        return self.__left_black

    @left_black.setter
    def left_black(self, left_black) -> None:
        self.__left_black = left_black

    @property
    def brightness(self):
        return self.__brightness

    @brightness.setter
    def brightness(self, brightness) -> None:
        self.__brightness = brightness


def find_right_black(image, black_bias=0):
    right_image = image[:240, 388:450]
    minimum = np.min(right_image)
    if minimum < 32862:
        return 32862
    return minimum + black_bias


def find_left_black(image, black_bias=0):
    left_image = image[:240, 30:120]
    minimum = np.min(left_image)
    if minimum < 32862:
        return 32862
    return minimum + black_bias


def find_left_centre(row: int, column: int,
                     bias_row: int = 0, bias_column: int = 0):
    left_centre_row = row + bias_row
    left_centre_column = column - 255 + bias_column
    return left_centre_row, left_centre_column


def mean_in_array(array):
    mean = array[np.nonzero(array)].mean()
    return mean


def right_array(image, row: int, column: int, radius: int, ratio: float, black):
    right_light_array = image[
                        row - radius: row + radius + 1,
                        column - radius: column + radius + 1]
    right_light_array -= black
    right_light_array = np.where(right_light_array < 0, 0, right_light_array)
    right_light_array_max = np.max(right_light_array)
    right_light_array = np.where(
        right_light_array > (right_light_array_max * ratio), right_light_array,
        0)
    return right_light_array


def left_array(image, row, column, right_light_array, radius: int, black):
    right_light_array_shape = np.where(right_light_array > 1, 1, 0)
    raw_left_light_array = image[
                           row - radius: row + radius + 1,
                           column - radius: column + radius + 1]
    left_light_array = raw_left_light_array * right_light_array_shape
    left_light_array -= black
    left_light_array = np.where(left_light_array < 0, 0, left_light_array)
    return left_light_array


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

    def potential_neurons(self) -> list:
        """
        Find the brightest points in the image as potential neurons. They will
        be further processed inside Neurons().
        Structure: list[row, column, brightness].
        """
        potential_neurons = []
        image_max = np.max(self.bit8)
        brightness_threshold = self.parameters.peak_ratio * image_max
        start = int(self.bit8.shape[1] / 2)
        end = int(self.bit8.shape[0])
        for column in range(start, end):
            for row in range(0, self.bit8.shape[1]):
                if self.bit8[row][column] <= brightness_threshold:
                    continue
                if surrender(self.bit8, row, column,
                             self.parameters.peak_circle):
                    # brightness = self.bit16[row][column]
                    # potential_neurons.append([row, column, brightness])
                    potential_neurons.append([row, column])
        print("在image的potential_neurons: ", potential_neurons)
        return potential_neurons

    def labelled(self, neurons: dict):
        # --- 设置一个默认的neurons？ ---
        labelled_image = self.image_bright(self.bit8,
                                           self.parameters.alpha,
                                           self.parameters.beta)

        for key in neurons:
            # draw rectangles and text for the right-half image
            centre = neurons.get(key)
            draw_rectangle(labelled_image, centre[0], centre[1], key,
                           self.parameters.label_radius)
            # draw rectangles and text for the left-half image
            left_row, left_column \
                = find_left_centre(centre[0], centre[1],
                                   self.parameters.row_bias,
                                   self.parameters.column_bias)
            draw_rectangle(labelled_image, left_row, left_column, key,
                           self.parameters.label_radius)
        return labelled_image

    def inform(self, neurons: dict) -> ImageInform:
        print("inform中: ")
        # inform on the right half
        max_brightness = 0
        max_row = 0
        max_column = 0
        print("neurons: ", neurons)
        if neurons == {}:
            return ImageInform()
        for key in neurons:
            centre = neurons.get(key)
            print("找到centre: ", self.bit8[centre[0]][centre[1]])
            if self.bit8[centre[0]][centre[1]] > max_brightness:
                max_brightness = self.bit8[centre[0]][centre[1]]
                max_row = centre[0]
                max_column = centre[1]
        right_black = find_right_black(self.bit16,
                                       self.parameters.right_black_bias)
        right_light_array = right_array(self.bit16, max_row, max_column,
                                        self.parameters.right_circle,
                                        self.parameters.right_ratio,
                                        right_black)
        right_brightness = mean_in_array(right_light_array)
        # inform on the left half
        left_row, left_column \
            = find_left_centre(max_row, max_column,
                               self.parameters.row_bias,
                               self.parameters.column_bias)
        left_black = find_left_black(self.bit16,
                                     self.parameters.left_black_bias)
        left_light_array = left_array(self.bit16, left_row, left_column,
                                      right_light_array,
                                      self.parameters.right_circle, left_black)
        left_brightness = mean_in_array(left_light_array)
        # generate image inform
        inform = ImageInform(self.num,
                             max_row, max_column,
                             right_brightness, right_black,
                             left_row, left_column,
                             left_brightness, left_black)
        return inform

    def image_bright(self, image, alpha=3, beta=0):
        image_bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image_bright
