import os

from PySide2 import QtCore, QtGui
import pandas as pd
from PySide2.QtCore import *

import cv2
import numpy as np
import time

from neurons import Neurons, NeuronData
from parameters import Parameters
from image import Image, ImageInform
from algorithms import Assignment


def cv_to_qpix(img):
    # cv 图片转换成 qpix图片
    qt_img = QtGui.QImage(img.data,  # 数据源
                          img.shape[1],  # 宽度
                          img.shape[0],  # 高度
                          img.shape[1],  # 行字节数
                          QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qt_img)


def helper(points: list) -> dict:
    # just helper, delete later
    result = {}
    i = 0
    for item in points:
        result[str(i)] = item
        i += 1
    return result


class ImageProcessingThread(QObject):
    """
    This class opens the thread at the back end, responding to image processing.
    """
    # 该类是作为程序后端负责打开并且处理图像的线程
    # data signal of image
    # 处理完毕图像数据信号
    show_img_signal = QtCore.Signal(QtGui.QPixmap, ImageInform)
    show_img_signal_loop = QtCore.Signal(QtGui.QPixmap, ImageInform)
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
        self.neuron_data = NeuronData()

    def loop(self, parameters, image_num, image_path, flip, start, end):
        for i in range(start, end + 1):
            print("进入时的image num：", image_num)
            print(image_path)
            self.image_processing_loop(parameters, image_num, image_path, flip)
            image_num += 1
            print("image processing之后再加一的image num：", image_num)
            image_path, _ = os.path.split(image_path)
            path = os.path.join(image_path, str(image_num) + ".tif")
            if not os.path.exists(path):
                image_path = os.path.join(image_path,
                                          f'{image_num:04}' + '.tif')
            else:
                image_path = path
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
        # neurons, including the assignment algorithm | 生成Neurons（包含了匹配算法）
        # backup | neurons的备份代码:
        # neurons = helper(image.potential_neurons())
        neurons = Neurons(image_num,
                          self.neuron_data.get(image_num),
                          self.neuron_data.position_header,
                          self.neuron_data.amount, image.potential_neurons())
        if self.neuron_data.is_min_image_num(image_num):
            neurons.assigned = neurons.to_dict()
            self.neuron_data.amount = len(neurons.assigned)
            self.neuron_data.add_neurons(neurons, 0)
        elif self.neuron_data.is_second_min_image_num(image_num):
            assignment = Assignment(self.neuron_data.amount, neurons,
                                    self.neuron_data.get_neurons(image_num - 1),
                                    -1)

            neurons.assigned = assignment.results()
            self.neuron_data.add_neurons(neurons, 1)
        else:
            assignment = Assignment(self.neuron_data.amount, neurons,
                                    self.neuron_data.get_neurons(image_num - 1),
                                    self.neuron_data.get_neurons(image_num - 2))
            neurons.assigned = assignment.results()
            self.neuron_data.add_neurons(neurons)

        # update this-image inform with calculated neurons | 更新图片信息
        img_inform = image.inform(neurons.assigned)
        # add this information into save list | 将该image对应的信息加入保存列表
        self.neuron_data.add_neurons(neurons)
        self.neuron_data.add_data(img_inform)

        labelled_img = image.labelled(neurons.assigned)
        q_pixmap = cv_to_qpix(labelled_img)
        self.show_img_signal_loop.emit(q_pixmap, img_inform)

    def image_processing(self, parameters, image_num, image_path, flip):
        """!!!这里应该清空之前的neuron记录！！！"""
        # 用于显示当前image的信息（在循环外）
        # 该方法连接了前端（作为槽函数
        # 初始化neuron data
        self.neuron_data = NeuronData()

        image = Image(image_path, image_num, parameters, flip)
        # neurons, including the assignment algorithm | 生成Neurons（包含了匹配算法）
        # backup | neurons的备份代码:
        # neurons = helper(image.potential_neurons())
        neurons = Neurons(image_num,
                          self.neuron_data.get(image_num),
                          self.neuron_data.position_header,
                          self.neuron_data.amount, image.potential_neurons())
        if self.neuron_data.is_min_image_num(image_num):
            neurons.assigned = neurons.to_dict()
            self.neuron_data.amount = len(neurons.assigned)
            self.neuron_data.add_neurons(neurons, 0)
        elif self.neuron_data.is_second_min_image_num(image_num):
            assignment = Assignment(self.neuron_data.amount, neurons,
                                    self.neuron_data.get_neurons(image_num - 1),
                                    -1)
            neurons.assigned = assignment.results()
            self.neuron_data.add_neurons(neurons, 1)
        else:
            assignment = Assignment(self.neuron_data.amount, neurons,
                                    self.neuron_data.get_neurons(image_num - 1),
                                    self.neuron_data.get_neurons(image_num - 2))
            neurons.assigned = assignment.results()
            self.neuron_data.add_neurons(neurons)
        # update this-image inform with calculated neurons | 更新图片信息
        img_inform = image.inform(neurons.assigned)
        # add this information into save list | 将该image对应的信息加入保存列表
        self.neuron_data.add_neurons(neurons)
        self.neuron_data.add_data(img_inform)

        labelled_img = image.labelled(neurons.assigned)
        q_pixmap = cv_to_qpix(labelled_img)
        self.show_img_signal.emit(q_pixmap, img_inform)

