import csv

from image import ImageInform, Image
from exceptions import OpenFileError


class Neurons(object):
    # 要导入csv对应的position信息（如果为空，则只分析图片）
    # 对应image的序号
    # >>> 重要：要收取ui的信号，获取当前应该分析的neuron的个数
    # 按照筛选层次定义methods（读取亮点，比较周围亮点，neurons指定数量的筛选，现在的相似三角形算法）
    # 算法现有问题：assign（等实现后比较结果），如果小于neurons数量怎么定
    def __init__(self, image_num: int,
                 position_data: list, position_header: list,
                 amount: int, potential: list) -> None:
        self.image_num = image_num  # may not start from 0
        self.position = position_data if position_data else [0, 0, 0, 0]  # 如果为空怎么处理，格式不对怎么办
        self.header = position_header if position_header else 1
        self.__amount = amount

        self.__potential: list = potential  # 不应该要potential，应该直接给结果；potential留给assign时接收
        self.physical_map_image = {}  # a map to find image position by physical position
        self.__assigned: dict = {}  # 重要：假设assigned之后都是储存的物理信息

    def physical_positions(self) -> list:
        # influenced by original pixel size, the size of image (1200x1200),
        # and transferred image (512x512)
        # 没有stage这些信息怎么处理？
        result = []
        trans_ratio = self.header if self.header == 1 else self.header[2]
        stage_x = self.position[0]
        stage_y = self.position[1]
        # notice: for stage, x is left-right, y is front-behind;
        # but for image coordinate, vice versa
        for neuron in self.potential:
            neuron_x = neuron[0]
            neuron_y = neuron[1]
            pp = [stage_x - (neuron_y - 512 / 4 * 3) * trans_ratio,
                  stage_y + (512 / 2 - neuron_x) * trans_ratio]
            self.physical_map_image[pp] = neuron
            result.append(pp)
        return result

    def to_dict(self) -> dict:
        """仅为第一张图片使用"""
        # 待修正： 后期考虑结合amount修正neurons
        result = {}
        i = 0
        for item in self.__potential:
            result[str(i)] = item
            i += 1
        # 更新信息
        self.__amount = len(result)
        self.assigned = result
        return result

    @property
    def potential(self) -> list:
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
        return the assigned and ordered neurons of the (image_num)th image

        :return: the assigned neurons
        """
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
    # （后端类实例需要一个self.data，在选择position file的button函数中需要传递路径）
    # 用于储存position信息（通过导入的csv文件；若无则空），
    # 处理csv的相关操作（给出对应序号图片的位置信息，读写csv），
    # 保存分析时的data结果（不保存图片，只保存Neurons；结果用于写入csv）（注意和现有ui的results有什么区别）
    # 最后通过stop按键触发保存
    def __init__(self) -> None:
        self.__position_path = ""
        # header: (追踪中心x，追踪中心y，pixel to len转换比例)
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
        # 待补充：检测不是csv文件，或者内容有误等情况的完善
        if self.__position_path == "":
            raise OpenFileError("Position file not selected or path not exist")
        self.__position_path = position_path
        # 准备将csv位置信息内容写入实例
        with open(position_path) as file:
            reader = list(csv.reader(file))
            try:
                self.position_header = reader[0]
                # 如果position文件的格式改变，需要更改这里的数字1
                self.__positions = reader[1:]
            except:
                print(f"Wrong format of csv file or empty csv file.")
                self.position_header = []
                # 如果position文件的格式改变，需要更改这里的数字1
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
        # 待补充：目前data采用key为0的neuron的数据
        """
        Add the given image information into saves for future save or
        modification. If this information has been stored (check by image
        number), replace the previous one. Otherwise, append the list.

        :param image_num:
        :param img_inform: the given image information
        """
        self.saves[image_num] = img_inform

    def save_data(self, images: dict[int, Image], save_path: str = '') -> None:
        # 待补充：尚未链接信号与槽函数（预计在ui的save data里面
        # 注意：现在只是正常运行程序，这个方法对应的结果还未测试
        if save_path == '':
            raise OpenFileError("Sava data path is empty")
        # write headers
        with open(save_path, 'w', encoding="utf-8", newline="") as file:
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
                if position not in neurons.potential:
                    position = neurons.physical_map_image[position]
                row = position[0]
                column = position[1]
                brightness = image.bit16[row][column]
                data.append(row)
                data.append(column)
                data.append(brightness)

            # open file and write in
            with open(save_path, 'w', encoding="utf-8", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)

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

        :param given_num:
        :return:
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
        neuron = self.neurons_save[image_num]
        position1 = neuron.assigned[neuron_num1]
        position2 = neuron.assigned[neuron_num2]
        # swap
        neuron.assigned[neuron_num1] = position2
        neuron.assigned[neuron_num2] = position1
        self.neurons_save[image_num] = neuron
