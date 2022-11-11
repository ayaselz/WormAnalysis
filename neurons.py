"""
This file includes classes Neurons and NeuronData. Neurons is a class to process
coordinate transformation in one specific image, save recognized highlights
from Image(), keep the result of tracking algorithm from Assignment() and
bridges tracking algorithms to the other part of the software. NeuronData is a
global class initialized at the beginning of back end, saving physical
position info (if exists), image info for images in form of ImageInform and
data of Neurons().

该文件包括类 Neurons 和 NeuronData。 Neurons()是一个类，用于处理一个特定图像中的坐标
变换、储存Image() 识别出的亮点和Assignment() 的跟踪算法的结果、并将跟踪算法连接到软件
的其他部分。 NeuronData 是一个在后端开始初始化的全局类，用于保存物理位置信息（如果存
在）、ImageInform 形式的图像的图像信息和 Neurons() 的数据。
"""

import csv

from image import ImageInform, Image
from exceptions import OpenFileError


class Neurons(object):
    """
    Neuron positions in a single image.
    """
    def __init__(self, image_num: int,
                 position_data: list, position_header: list,
                 amount: int, potential: list) -> None:
        """
        Constructor of Neurons
        :param image_num: the number of image
        :param position_data: physical position data read from CSV file;
            [0, 0, 0, 0] if none is read
        :param position_header: head data read from CSV file; 1 if none is read
        :param amount: the number of neurons in this image
        :param potential: recognized highlights from Image(), working as
        potential neurons
        """
        self.image_num = image_num
        self.position = position_data if position_data else [0, 0, 0, 0]
        self.header = position_header if position_header else 1
        self.__amount = amount
        self.__potential: list = potential
        # a map to find image position by physical position
        self.physical_map_image = {}
        # keep the result of tracking algorithm from Assignment()
        self.__assigned: dict = {}

    def physical_positions(self) -> list:
        """
        Transform the pixel-based neuron position into physical positions.
        :return: a list of transformed potential neurons
        """
        # notes: might be influenced by original pixel size, the size of
        # image (currently 1200x1200),
        result = []
        trans_ratio = self.header if self.header == 1 else self.header[2]
        stage_x = self.position[0]
        stage_y = self.position[1]
        # notice: for stage, x is left-right, y is front-behind;
        # but for image coordinate, vice versa
        for neuron in self.potential:
            neuron_x = neuron[0]
            neuron_y = neuron[1]
            pp = [stage_x - (neuron_y - 1200 / 4 * 3) * trans_ratio,
                  stage_y + (1200 / 2 - neuron_x) * trans_ratio]
            self.physical_map_image[pp] = neuron
            result.append(pp)
        return result

    def to_dict(self) -> dict:
        """
        Process the assignment if the number of image is 1
        :return: dict[neuron_num: str, coordinate: ndarray]
        """
        result = {}
        i = 0
        for item in self.__potential:
            result[str(i)] = item
            i += 1
        # update the info
        # 更新信息
        self.__amount = len(result)
        self.assigned = result
        return result

    @property
    def potential(self) -> list:
        """
        The getter of highlights from Image(). Physical positions are
        automatically implemented.
        :return: potential neurons
        """
        # if physical inform is given, return actual coordinates
        # otherwise picture position only
        if self.position == [0, 0, 0, 0] and self.header == 1:
            return self.__potential
        return self.physical_positions()

    @potential.setter
    def potential(self, potential: list) -> None:
        self.__potential = potential

    @property
    def assigned(self) -> dict:
        """
        Return the assigned and ordered neurons of the (image_num)th image

        :return: the assigned neurons
        """
        # completeness
        # 完备性
        if len(self.__assigned) < self.__amount:
            for i in range(self.__amount):
                if str(i) not in self.__assigned.keys():
                    self.__assigned[str(i)] = [-1, -1]
        # 差个逻辑： 开始时没有neuron怎么办
        return self.__assigned

    @assigned.setter
    def assigned(self, assigned: dict) -> None:
        self.__assigned = assigned


class NeuronData(object):
    """
    Class containing Neuron and ImageInform.
    """
    # （后端类实例需要一个self.data，在选择position file的button函数中需要传递路径）
    # 用于储存position信息（通过导入的csv文件；若无则空），
    # 处理csv的相关操作（给出对应序号图片的位置信息，读写csv），
    # 保存分析时的data结果（不保存图片，只保存Neurons；结果用于写入csv）（注意和现有ui的results有什么区别）
    # 最后通过stop按键触发保存
    def __init__(self) -> None:
        """
        NeuronData constructor
        """
        self.__position_path = ""
        self.position_header = []
        self.__positions = []
        self.__saves: dict[int, ImageInform] = {}
        self.neurons_save: dict[int, Neurons] = {}
        # the amount of tracked neurons
        self.__amount = 1

    @property
    def position_path(self) -> str:
        """
        the path of position file (.csv)

        :return:
        """
        return self.__position_path

    @position_path.setter
    def position_path(self, position_path: str) -> None:
        # check the existence of file
        if self.__position_path == "":
            raise OpenFileError("Position file not selected or path not exist")
        self.__position_path = position_path
        # read the CSV data
        # 准备将csv位置信息内容写入实例
        with open(position_path) as file:
            reader = list(csv.reader(file))
            try:
                self.position_header = reader[0]
                self.__positions = reader[1:]
            except:
                print(f"Wrong format of csv file or empty csv file.")
                self.position_header = []
                self.__positions = []

    def get(self, image_num: int) -> list:
        try:
            return self.__positions[image_num]
        except:
            print(f"This image does not have caught physical "
                  f"position data. Image number: {image_num}")
            return []

    @property
    def saves(self) -> dict[int, ImageInform]:
        return self.__saves

    @saves.setter
    def saves(self, saves: dict) -> None:
        self.__saves = saves

    def add_data(self, image_num: int, img_inform: ImageInform) -> None:
        """
        Add the given image information into saves for future save or
        modification. If this information has been stored (check by image
        number), replace the previous one. Otherwise, append the list.

        :param image_num: the number of image
        :param img_inform: the given image information
        """
        self.saves[image_num] = img_inform

    def save_data(self, images: dict[int, Image], save_path: str = '') -> None:
        """
        Executing at the end of analysis after clicking Stop button.
        """
        if save_path == '':
            raise OpenFileError("Sava data path is empty")
        # write headers
        with open(save_path, 'a', encoding="utf-8", newline='') as file:
            csv_writer = csv.writer(file)
            results_head = ['image_num',
                            'Right_row', 'Right_column', 'Right_brightness',
                            'Left_row', 'Left_column', 'Left_brightness',
                            'Brightness']
            for i in range(self.amount):
                results_head.append('Neuron_' + str(i) + '_row')
                results_head.append('Neuron_' + str(i) + '_column')
                results_head.append('Neuron_' + str(i) + '_brightness')
            csv_writer.writerow(results_head)
        # write in each line
        for key in self.saves:
            each = self.saves[key]
            data = [each.num,
                    each.right_row, each.right_column,
                    each.right_brightness,
                    each.left_row, each.left_column,
                    each.left_brightness,
                    each.brightness]
            # neuron position and brightness
            neurons = self.neurons_save[key]
            image = images[key]
            for tag in neurons.assigned:
                position = neurons.assigned[tag]
                # print(type(position), position)
                # print(type(neurons.potential), neurons.potential)

                # for item in neurons.potential:
                #     if item[0] == position[0] and item[1] == position[1]:
                #         break
                #     else:
                #         print(neurons.physical_map_image)
                #         position = neurons.physical_map_image[position]
                if type(position) != list:  # might be ndarray
                    position = position.tolist()
                if position not in neurons.potential and neurons.physical_map_image:
                    position = neurons.physical_map_image[position]

                row = position[0]
                column = position[1]
                brightness = image.bit16[row][column]
                data.append(row)
                data.append(column)
                data.append(brightness)

            # open file and write in
            with open(save_path, 'a', encoding="utf-8", newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)

        print("=== Data has been saved ===")

    # @property
    # def neurons_save(self) -> dict[int, Neurons]:
    #     return self.neurons_save
    #
    # @neurons_save.setter
    # def neurons_save(self, neurons_save: dict) -> None:
    #     self.neurons_save = neurons_save

    def add_neurons(self, image_num: int, neurons: Neurons) -> None:
        self.neurons_save[image_num] = neurons

    @property
    def amount(self) -> int:
        return self.__amount

    @amount.setter
    def amount(self, amount: int) -> None:
        self.__amount = amount

    def get_neurons(self, given_num: int) -> Neurons | int:
        """
        Return Neurons type with the given tag.

        :param given_num: the number of image
        :return: Neurons()
        """
        if given_num in self.neurons_save.keys():
            return self.neurons_save[given_num]
        return -1

    def is_min_image_num(self, image_num: int) -> bool:
        for key in self.neurons_save:
            if key < image_num:
                return False
        return True

    def swap(self, image_num: int, neuron_num1: str, neuron_num2: str) -> None:
        """
        Exchange the information in neurons_saves of the specified neurons
        in the specified image.
        :param image_num: the number of the specified image
        :param neuron_num1: the number of the specified neuron
        :param neuron_num2: the other number of the specified neuron
        :return:
        """
        neuron = self.neurons_save[image_num]
        position1 = neuron.assigned[neuron_num1]
        position2 = neuron.assigned[neuron_num2]
        # swap
        neuron.assigned[neuron_num1] = position2
        neuron.assigned[neuron_num2] = position1
        self.neurons_save[image_num] = neuron
