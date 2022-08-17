from PySide2 import QtCore, QtGui
import pandas as pd
from PySide2.QtCore import *

import cv2
import numpy as np
import time

from neurons import Neurons
from parameters import Parameters
from image import Image


class ImageProcessingThread(QObject):
    """
    This class opens the thread at the back end, responding to image processing.
    """
    # 该类是作为程序后端负责打开并且处理图像的线程
    # data signal of image
    # 处理完毕图像数据信号
    show_img_signal = QtCore.Signal(QtGui.QPixmap, dict)
    show_img_signal_loop = QtCore.Signal(QtGui.QPixmap, dict)
    #  signal of parameters
    # 线程接收参数信号
    start_image_process_thread_signal \
        = QtCore.Signal(Parameters, int, str, bool)
    loop_signal = QtCore.Signal(Parameters, int, str, bool, int, int)

    def __init__(self):
        super(ImageProcessingThread, self).__init__()
        # initialize signals for pausing and killing the loop
        self.is_paused = False
        self.is_killed = False

    def loop(self, parameters, image_num, image_path, flip, start, end):
        for i in range(start, end + 1):
            self.image_processing_loop(parameters, i, image_path, flip)
            # wait for 0.1 and then process the next image
            time.sleep(0.1)

            while self.is_paused:
                time.sleep(0.1)
            if self.is_killed:
                break

    def image_processing_loop(self, parameters: Parameters, image_num: int,
                              image_path: str, flip: bool) -> None:
        """
        Process the (image_num)th image in the Back-end loop.
        :param parameters:
        :param image_num:
        :param image_path:
        :param flip:
        :return:
        """
        image = Image(image_path, image_num, parameters, flip)
        image_16bit, image_8bit = image.bit16, image.bit8
        # 最后要得到Neurons的结果、image亮度（？亮的image），以方便后续
        result_dict, image_bright \
            = self.process_image(parameters, image_num, image_16bit, image_8bit)
        q_pixmap = self.cv_to_qpix(image_bright)

        self.show_img_signal_loop.emit(q_pixmap, result_dict)

    def image_processing(self, parameters, image_num, image_path, flip):
        # 该方法连接了前端（作为槽函数
        image = Image(image_path, image_num, parameters, flip)
        image_16bit, image_8bit = image.bit16, image.bit8
        result_dict, image_bright \
            = self.process_image(parameters, image_num, image_16bit, image_8bit)
        q_pixmap = self.cv_to_qpix(image_bright)

        self.show_img_signal.emit(q_pixmap, result_dict)

    def process_image(self, parameters, image_num, image_16bit, image_8bit):

        right_centres = self.find_peak_point(
            image_8bit, parameters.peak_circle, parameters.peak_ratio)
        # --- 将读取的亮点放入Neurons类进行加工 ---
        self.neurons.add_neuron(right_centres)
        right_centres = self.neurons.current_neuron()
        # --- end ---
        print("right centres: ", right_centres)
        image_bright = \
            self.image_bright(image_8bit, parameters.alpha, parameters.beta)

        image_bright = self.label(
            image_bright, right_centres,
            parameters.label_radius,
            parameters.row_bias,
            parameters.column_bias
        )
        max_brightness, max_row, max_column \
            = self.find_max_brightness(image_8bit, right_centres)
        right_black = self.right_black(image_16bit, parameters.right_black_bias)
        left_black = self.left_black(image_16bit, parameters.left_black_bias)

        result_dict = self.calculate_brightness(
            image_16bit, image_num, max_row, max_column,
            right_black, left_black,
            parameters.right_circle, parameters.right_ratio,
            parameters.row_bias,
            parameters.column_bias
        )
        result_dict['right_black'] = right_black
        result_dict['left_black'] = left_black

        return result_dict, image_bright

    def cv_to_qpix(self, img):
        # cv 图片转换成 qpix图片
        qt_img = QtGui.QImage(img.data,  # 数据源
                              img.shape[1],  # 宽度
                              img.shape[0],  # 高度
                              img.shape[1],  # 行字节数
                              QtGui.QImage.Format_Grayscale8)
        return QtGui.QPixmap.fromImage(qt_img)

    def image_bright(self, image, alpha=3, beta=0):
        image_bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image_bright

    def find_left_centre(self, column, row, bias_row=0, bias_column=0):
        left_centre_row = row + bias_row
        left_centre_column = column - 255 + bias_column
        return left_centre_row, left_centre_column

    def surrender_value_compare(self, image, row, column, surrender):
        value = image[row][column]
        for i in range(0, 2 * surrender + 1):
            if 0 < row - surrender + i < image.shape[0] and 0 < column - surrender \
                    and column + surrender < image.shape[1]:
                if value < image[row - surrender + i][column - surrender] \
                        or value < image[row - surrender + i][column + surrender]:
                    return False
        for i in range(1, 2 * surrender):
            if 0 < row - surrender and row + surrender < image.shape[0] \
                    and 0 < column - surrender + i < image.shape[1]:
                if value < image[row - surrender][column - surrender + i] \
                        or value < image[row + surrender][column - surrender + i]:
                    return False
        return True

    def surrender(self, image, row, column, circle):
        for i in range(1, int(circle) + 1):
            if not self.surrender_value_compare(image, row, column, i):
                return False
        return True

    def find_peak_point(self, image, circle=6, ratio=0.4):
        centres = []
        image_max = np.max(image)
        start = int(image.shape[1] / 2)
        end = int(image.shape[0])
        for column in range(start, end):
            for row in range(0, image.shape[1]):
                if image[row][column] > (ratio * image_max):
                    if self.surrender(image, row, column, circle):
                        centres.append([row, column])

        return centres

    def draw_rectangle(self, image, column, row, label_text, radius=5, text_place=5):
        # 框一半边长
        # 标签离框的距离
        cv2.rectangle(image, (int(column - radius), int(row - radius)),
                      (int(column + radius), int(row + radius)), 255)
        cv2.putText(image, label_text, (column - radius, row - radius - text_place),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)

    def right_black(self, image, black_bias=0):
        right_image = image[:240, 388:450]
        minimum = np.min(right_image)
        if minimum < 32862:
            return 32862
        return minimum + black_bias

    def left_black(self, image, black_bias=0):
        left_image = image[:240, 30:120]
        minimum = np.min(left_image)
        if minimum < 32862:
            return 32862
        return minimum + black_bias

    def find_max_brightness(self, image, centres):
        max_brightness = 0
        max_row = 0
        max_column = 0
        for centre in centres:
            if image[centre[0]][centre[1]] > max_brightness:
                max_brightness = image[centre[0]][centre[1]]
                max_row = centre[0]
                max_column = centre[1]

        return max_brightness, max_row, max_column

    def mean_in_array(self, array):
        mean = array[np.nonzero(array)].mean()
        return mean

    def right_array(self, image, row, column, radius, ratio, black):
        right_light_array = image[row - radius: row + radius + 1, column - radius: column + radius + 1]
        right_light_array -= black

        right_light_array = np.where(right_light_array < 0, 0, right_light_array)
        right_light_array_max = np.max(right_light_array)

        right_light_array = np.where(right_light_array > (right_light_array_max * ratio), right_light_array, 0)
        return right_light_array

    def left_array(self, image, left_centre_row, left_centre_column, right_light_array, radius, black):
        right_light_array_shape = np.where(right_light_array > 1, 1, 0)
        raw_left_light_array = image[left_centre_row - radius: left_centre_row + radius + 1,
                               left_centre_column - radius: left_centre_column + radius + 1]
        left_light_array = raw_left_light_array * right_light_array_shape
        left_light_array -= black
        left_light_array = np.where(left_light_array < 0, 0, left_light_array)
        return left_light_array

    def calculate_brightness(self, image, image_num, row, column, right_black, left_black, radius=5, ratio=0.6,
                             row_bias=0,
                             column_bias=0):
        left_row, left_column = self.find_left_centre(column, row, row_bias, column_bias)

        right_light_array = self.right_array(image, row, column, radius, ratio, right_black)

        print(right_light_array)
        left_light_array = \
            self.left_array(image, left_row, left_column, right_light_array, radius, left_black)

        print(left_light_array)
        right_brightness = self.mean_in_array(right_light_array)
        left_brightness = self.mean_in_array(left_light_array)
        result_dict = {
            'image_num': image_num,
            'right_row': row, 'right_column': column, 'right_brightness': right_brightness,
            'left_row': left_row, 'left_column': left_column, 'left_brightness': left_brightness,
            'brightness': left_brightness / right_brightness
        }
        return result_dict

    def write_csv(self, result_dict, dataframe):
        dataframe = \
            dataframe.append(pd.DataFrame({
                'Right_row': [result_dict['right_row']],
                'Right_column': [result_dict['right_column']],
                'Right_brightness': [result_dict['right_brightness']],
                'Left_row': [result_dict['left_row']],
                'Left_column': [result_dict['left_column']],
                'Left_brightness': [result_dict['left_brightness']],
                'Brightness': [result_dict['brightness']]}),
                ignore_index=True)
        return dataframe

    def rotate_picture(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(image, m, (w, h))
        return rotated

    def label(self, image, centres, label_radius, bias_row, bias_column):
        for key in self.neurons.get_neurons():
            centre = self.neurons.get_neurons().get(key)[-1]
            self.draw_rectangle(image, centre[1], centre[0], key, label_radius)

            left_row, left_column = self.find_left_centre(centre[1], centre[0], bias_row, bias_column)
            self.draw_rectangle(image, left_column, left_row, key, label_radius)
        return image
        # for centre in centres:
        #     label_text = str(centres.index(centre))
        #     self.draw_rectangle(image, centre[1], centre[0], label_text, label_radius)
        #
        #     left_row, left_column = self.find_left_centre(centre[1], centre[0], bias_row, bias_column)
        #     self.draw_rectangle(image, left_column, left_row, label_text, label_radius)
        #     break
        # return image