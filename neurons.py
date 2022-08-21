import csv

from image import ImageInform
from exceptions import OpenFileError


# def assign(neurons: dict, neuron: list):
#     """Put a series neuron in the Neurons dict in order"""
#     for i in range(len(neuron)):
#         neurons[str(i)].append(neuron[i])
#
#
# def match_score(previous, current):
#     """当只有两个neuron时，根据欧式距离得出分数"""
#     return (previous[0] - current[0]) ** 2 + (previous[1] - current[1]) ** 2
#
#
# class Neurons(object):
#     # 神经元数量，读取csv（对应行的位置信息
#     def __init__(self):
#         self.neurons = {"0": [],
#                         "1": []}
#
#     def get_neurons(self) -> dict:
#         return self.neurons.copy()
#
#     def current_neuron(self) -> list:
#         """The position of current neuron on the image. """
#         # 改成multi-neuron
#         return [self.get_neurons()["0"][-1], self.get_neurons()["1"][-1]]
#
#     def add_neuron(self, neuron: list):
#         if self.neurons["0"]:
#             # 执行匹配算法
#             self.match(neuron)
#         else:
#             assign(self.neurons, neuron)
#
#     def match(self, neuron: list):
#         """执行算法（先尝试简单的距离尝试）"""
#         previous = self.current_neuron()
#         print("previous neuron list:", previous)
#         for i in range(len(previous)):
#             position = previous[i]
#             # 确保position取值不是[-1, -1]
#             while position == [-1, -1]:
#                 j = -2
#                 position = self.neurons.get(str(i))[j]
#                 j -= 1
#             print(position)
#             current_score = None
#             candidate = []
#             for item in neuron:
#                 match = match_score(position, item)
#                 print("the score of", item, "is", match)
#                 if (current_score is None) or (match < current_score):
#                     current_score = match
#                     candidate = item
#
#             # neuron.删除candidate
#             print("Neurons_match:", candidate, "is the candidate of", position)
#             # 下面检查是否有任意candidate为空列表。
#             # 当存在candidate为空的时候，说明读取到的亮点数少于实际神经元数。
#             # 这种情况下，将该candidate坐标标记为[-1, -1]以示异常
#             if not candidate:
#                 candidate = [-1, -1]
#             else:
#                 # 检查没问题的情况下，再remove
#                 neuron.remove(candidate)
#             # matching得到的位置
#             self.neurons[str(i)].append(candidate)


class Neurons(object):
    # 要导入csv对应的position信息（如果为空，则只分析图片）
    # 对应image的序号
    # >>> 重要：要收取ui的信号，获取当前应该分析的neuron的个数
    # 按照筛选层次定义methods（读取亮点，比较周围亮点，neurons指定数量的筛选，现在的相似三角形算法）
    # 算法现有问题：assign（等实现后比较结果），如果小于neurons数量怎么定
    def __init__(self, image_num: int,
                 position_data: list, position_header: list,
                 amount: int, potential: list) -> None:
        self.__image_num = image_num  # start from 0
        self.position = position_data if position_data else [0, 0, 0,
                                                             0]  # 如果为空怎么处理，格式不对怎么办
        self.header = position_header if position_header else 1
        self.__amount = amount
        self.potential = potential  # 不应该要potential，应该直接给结果；potential留给assign时接收
        self.__assigned = {}  # 重要：假设assigned之后都是储存的物理信息

    def physical_positions(self, neurons: list) -> list:
        # influenced by original pixel size, the size of image (1200x1200),
        # and transferred image (512x512)
        # 没有stage这些信息怎么处理？
        result = []
        trans_ratio = self.header * (1200 / 512)
        stage_x = self.position[0]
        stage_y = self.position[1]
        # notice: for stage, x is left-right, y is front-behind;
        # but for image coordinate, vice versa
        for neuron in neurons:
            neuron_x = neuron[0]
            neuron_y = neuron[1]
            pp = [stage_x - (neuron_y - 512 / 4 * 3) * trans_ratio,
                  stage_y + (512 / 2 - neuron_x) * trans_ratio]
            result.append(pp)
        return result

    def to_dict(self) -> dict:
        # 待修正： 后期考虑结合amount修正neurons
        result = {}
        i = 0
        for item in self.__potential:
            result[str(i)] = item
            i += 1
        return result

    @property
    def assigned(self) -> dict:
        """
        return the assigned and ordered neurons of the (image_num)th image

        :return: a copy of the assigned neurons
        """
        # 差个逻辑： 开始时没有neuron怎么办
        return self.__assigned.copy()

    @assigned.setter
    def assigned(self, assigned: dict) -> None:
        self.__assigned = assigned

    @property
    def image_num(self) -> int:
        return self.__image_num

    @image_num.setter
    def image_num(self, image_num: int) -> None:
        self.__image_num = image_num


class NeuronData(object):
    # （后端类实例需要一个self.data，在选择position file的button函数中需要传递路径）
    # 用于储存position信息（通过导入的csv文件；若无则空），
    # 处理csv的相关操作（给出对应序号图片的位置信息，读写csv），
    # 保存分析时的data结果（不保存图片，只保存Neurons；结果用于写入csv）（注意和现有ui的results有什么区别）
    # 最后通过stop按键触发保存
    def __init__(self) -> None:
        self.__position_path = ""
        # header: (追踪中心x，追踪中心y，pixel to len转换比例)
        self.__position_header = []
        self.__positions = []
        self.__saves = []
        self.__neurons_save = []
        # the amount of tracked neurons
        self.__amount = 1

    @property
    def position_path(self) -> str:
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
            self.__position_header = reader[0]
            # 如果position文件的格式改变，需要更改这里的数字1
            self.__positions = reader[1:]

    @property
    def header(self) -> list:
        return self.__position_header.copy()

    @header.setter
    def header(self, header: list) -> None:
        self.__position_header = header

    def get(self, image_num: int) -> list:
        # 待修正：比例转换之后再return，先需要知道stage返回的单位
        # 待补充：检测position格式是否有误再返回
        # position = self.__positions[image_num]
        # trans_ratio = self.header[2]
        # return [position[0], position[1],
        #         position[2] * trans_ratio, position[3] * trans_ratio]
        return self.__positions[image_num]

    @property
    def saves(self) -> list:
        return self.__saves.copy()

    @saves.setter
    def saves(self, saves: list) -> None:
        self.__saves = saves

    def add_data(self, img_inform: ImageInform) -> None:
        # 待补充：目前data采用key为0的neuron的数据
        """
        Add the given image information into saves for future save or
        modification. If this information has been stored (check by image
        number), replace the previous one. Otherwise, append the list.

        :param img_inform: the given image information
        """
        for data in self.__saves:
            if img_inform.num == data.num:
                index = self.__saves.index(data)
                self.__saves[index] = img_inform
            else:
                self.__saves.append(img_inform)

    def save_data(self, save_path: str = '') -> None:
        # 待补充：尚未链接信号与槽函数（预计在ui的save data里面
        # 注意：现在只是正常运行程序，这个方法对应的结果还未测试
        if save_path == '':
            raise OpenFileError("Sava data path is empty")
        with open(save_path, 'w', encoding="utf-8", newline="") as file:
            csv_writer = csv.writer(file)
            results_head = ['image_num',
                            'Right_row', 'Right_column', 'Right_brightness',
                            'Left_row', 'Left_column', 'Left_brightness',
                            'Brightness']
            csv_writer.writerow(results_head)
            for each in self.saves:
                data = [each.num,
                        each.right_row, each.right_column,
                        each.right_brightness,
                        each.left_row, each.left_column,
                        each.left_brightness,
                        each.brightness]
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)

    @property
    def neurons_save(self) -> list:
        return self.__neurons_save.copy()

    @neurons_save.setter
    def neurons_save(self, neurons_save: list) -> None:
        self.__neurons_save = neurons_save

    def add_neurons(self, neurons: Neurons) -> None:
        # 待补充：如果保证没有重复neurons数据
        self.__neurons_save.append(neurons)

    @property
    def amount(self) -> int:
        return self.__amount

    @amount.setter
    def amount(self, amount: int) -> None:
        self.__amount = amount

    def is_min_image_num(self, given_num) -> bool:
        if not self.__neurons_save:
            return True
        for neurons in self.__neurons_save:
            if neurons.image_num < given_num:
                return False
        return True

    def get_neurons(self, given_num: int) -> Neurons | None:
        for neurons in self.__neurons_save:
            if neurons.image_num == given_num:
                return neurons
        return None
