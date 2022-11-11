"""
The main class for back end.
"""

import os

from PySide2 import QtCore, QtGui
import pandas as pd
from PySide2.QtCore import *

import cv2
import numpy as np
import time
import warnings

from neurons import Neurons, NeuronData
from parameters import Parameters
from image import Image, ImageInform
from algorithms import Assignment


def cv_to_qpix(img):
    """
    Transfer from CV image to qpix image
    """
    qt_img = QtGui.QImage(img.data,  # the source
                          img.shape[1],  # width
                          img.shape[0],  # height
                          img.shape[1],  # row bytes
                          QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qt_img)


def helper(points: list) -> dict:
    # just helper, might be deleted later
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
        self.assignment = Assignment()
        self.images: dict[int, Image] = {}

    def loop(self, parameters, image_num, image_path, flip, start, end):
        """
        It is triggered after clicking Run button on GUI
        :param parameters: set parameters passed from GUI
        :param image_num: the number of starting image
        :param image_path: the path of the image file
        :param flip: the status of image flip
        :param start: the starting image number
        :param end: the ending image number
        """
        start_time = time.time()

        for i in range(start, end + 1):
            # process the image
            self.image_processing_loop(parameters, image_num, image_path, flip)
            # move to the next image when the previous process is done
            image_num += 1
            # reset the file path of image
            image_path, _ = os.path.split(image_path)
            path = os.path.join(image_path, str(image_num) + ".tif")
            if not os.path.exists(path):
                image_path = os.path.join(image_path,
                                          f'{image_num:04}' + '.tif')
            else:
                image_path = path
            # wait for 0.1 and then process the next image
            time.sleep(0.1)
            # sleep the loop after clicking Pause
            while self.is_paused:
                time.sleep(0.1)
            # end the loop after clicking Stop
            if self.is_killed:
                break

        stop_time = time.time()
        print('Runtime: ', (stop_time - start_time) * 10**3, "ms")

    def image_processing_loop(self, parameters: Parameters, image_num: int,
                              image_path: str, flip: bool) -> None:
        """
        Process the (image_num)th image in the Back-end loop.

        :param parameters: the set parameters from GUI
        :param image_num: the number of current image
        :param image_path: the file path of current image
        :param flip: the status of image flip
        """
        # 1. receive and transfer image data
        image = Image(image_path, image_num, parameters, flip)
        # put this image in the images list
        self.images[image_num] = image
        if self.neuron_data.is_min_image_num(image_num):
            # if is the first image in the process
            # find the amount of neurons
            amount = len(image.potential_neurons())
            # set the neuron amount in NeuronData() and Assignment
            self.neuron_data.amount = amount
            self.assignment.amount = amount
        # 2. generate Neurons of this image and recognized highlights
        neurons = Neurons(image_num,
                          self.neuron_data.get(image_num),
                          self.neuron_data.position_header,
                          self.neuron_data.amount,
                          image.potential_neurons())
        if self.neuron_data.is_min_image_num(image_num):
            # if this is the first image, highlights are regarded as the neurons
            neurons.assigned = neurons.to_dict()
            # the neurons positions are added to Assignment()
            self.assignment.add_neurons(image_num, neurons.assigned)
        else:
            # 3. Calculate the neuron positions from the potential highlights
            self.assignment.assign(image_num, neurons.potential)
            neurons.assigned = self.assignment.get_neurons(image_num)

        # update this-image inform with calculated neurons | 更新图片信息
        img_inform = image.inform(neurons.assigned)
        # add this information into save list | 将该image对应的信息加入保存列表
        self.neuron_data.add_neurons(image_num, neurons)
        self.neuron_data.add_data(image_num, img_inform)
        # show labelled image on GUI | 将画了标签的图片显示在UI上面
        labelled_img = image.labelled(neurons.assigned)
        q_pixmap = cv_to_qpix(labelled_img)
        self.show_img_signal_loop.emit(q_pixmap, img_inform)

    def image_processing(self, parameters, image_num, image_path, flip):
        """This is used to show data of single image on GUI"""
        # initialize neuron data, assignment and images in case the future
        # adjustment: jump to specified image
        self.neuron_data = NeuronData()
        self.assignment = Assignment()
        self.images = {}

        self.image_processing_loop(parameters, image_num, image_path, flip)

    def swap_neuron_position(self, image_num: int, tag1: str, tag2: str):
        """Exchange the information of the specified neurons in the specified
        image, including NeuronData and Assignment."""
        if self.is_paused:
            # swap the data (swap the existing image)
            print("before swap: ", self.neuron_data.get_neurons(image_num).assigned)
            self.neuron_data.swap(image_num, tag1, tag2)
            self.assignment.add_neurons(image_num, self.neuron_data.get_neurons(image_num).assigned)
            print("Swap: success!")
            # reload the image with swapped data
            image = self.images[image_num]
            neurons = self.neuron_data.get_neurons(image_num)
            print("after swap: ", self.neuron_data.get_neurons(image_num).assigned)
            labelled_img = image.labelled(neurons.assigned)
            img_inform = self.images[image_num].inform(neurons.assigned)
            q_pixmap = cv_to_qpix(labelled_img)
            self.show_img_signal_loop.emit(q_pixmap, img_inform)

        else:
            warnings.warn("Please swap during pause.")
