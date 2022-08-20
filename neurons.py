import csv


def assign(neurons: dict, neuron: list):
    """Put a series neuron in the Neurons dict in order"""
    for i in range(len(neuron)):
        neurons[str(i)].append(neuron[i])


def match_score(previous, current):
    """当只有两个neuron时，根据欧式距离得出分数"""
    return (previous[0] - current[0]) ** 2 + (previous[1] - current[1]) ** 2


class Neurons(object):
    # 神经元数量，读取csv（对应行的位置信息
    def __init__(self):
        self.neurons = {"0": [],
                        "1": []}

    def get_neurons(self) -> dict:
        return self.neurons.copy()

    def current_neuron(self) -> list:
        """The position of current neuron on the image. """
        # 改成multi-neuron
        return [self.get_neurons()["0"][-1], self.get_neurons()["1"][-1]]

    def add_neuron(self, neuron: list):
        if self.neurons["0"]:
            # 执行匹配算法
            self.match(neuron)
        else:
            assign(self.neurons, neuron)

    def match(self, neuron: list):
        """执行算法（先尝试简单的距离尝试）"""
        previous = self.current_neuron()
        print("previous neuron list:", previous)
        for i in range(len(previous)):
            position = previous[i]
            # 确保position取值不是[-1, -1]
            while position == [-1, -1]:
                j = -2
                position = self.neurons.get(str(i))[j]
                j -= 1
            print(position)
            current_score = None
            candidate = []
            for item in neuron:
                match = match_score(position, item)
                print("the score of", item, "is", match)
                if (current_score is None) or (match < current_score):
                    current_score = match
                    candidate = item

            # neuron.删除candidate
            print("Neurons_match:", candidate, "is the candidate of", position)
            # 下面检查是否有任意candidate为空列表。
            # 当存在candidate为空的时候，说明读取到的亮点数少于实际神经元数。
            # 这种情况下，将该candidate坐标标记为[-1, -1]以示异常
            if not candidate:
                candidate = [-1, -1]
            else:
                # 检查没问题的情况下，再remove
                neuron.remove(candidate)
            # matching得到的位置
            self.neurons[str(i)].append(candidate)


class Neurons(object):
    # 要导入csv对应的position信息（如果为空，则只分析图片）
    # 对应image的序号
    # 重要：要收取ui的信号，获取当前应该分析的neuron的个数
    # 按照筛选层次定义methods（读取亮点，比较周围亮点，neurons指定数量的筛选，现在的相似三角形算法）
    # 算法现有问题：assign（等实现后比较结果），如果小于neurons数量怎么定
    def __init__(self, image_num: int = 2) -> None:
        self.image_num = image_num
        # self.points = points


class NeuronData(object):
    # （后端类实例需要一个self.data，在选择position file的button函数中需要传递路径）
    # 用于储存position信息（通过导入的csv文件；若无则空），
    # 处理csv的相关操作（给出对应序号图片的位置信息，读写csv），
    # 保存分析时的data结果（不保存图片，只保存Neurons；结果用于写入csv）
    def __init__(self) -> None:
        self.__position_path = ""
        # header: (追踪中心x，追踪中心y，pixel to len转换比例)
        self.__position_header = []
        self.__positions = []

    @property
    def position_path(self) -> str:
        return self.__position_path

    @position_path.setter
    def position_path(self, position_path: str) -> None:
        # 待补充：检测不是csv文件，或者内容有误等情况的完善
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

    def get(self, image_num:  int) -> list:
        # 待修正：比例转换之后再return，先需要知道stage返回的单位
        # 待补充：检测position格式是否有误再返回
        # position = self.__positions[image_num]
        # trans_ratio = self.header[2]
        # return [position[0], position[1],
        #         position[2] * trans_ratio, position[3] * trans_ratio]
        return self.__positions[image_num]
