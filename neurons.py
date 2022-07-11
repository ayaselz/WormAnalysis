import numpy as np


def assign(neurons: dict, neuron: list):
    """Put a series neuron in the Neurons dict in order"""
    for i in range(len(neuron)):
        neurons[str(i)].append(neuron[i])


def match_score(previous, current):
    print("trigger: function match_score")
    """当只有两个neuron时，根据欧式距离得出分数"""
    return np.sqrt((previous[0] - current[0]) ** 2
                   + (previous[1] - current[1]) ** 2)


class Neurons(object):
    def __init__(self):
        print("class Neurons generated")
        self.neurons = {"0": [],
                        "1": []}

    def get_neurons(self) -> dict:
        return self.neurons.copy()

    def current_neuron(self) -> list:
        """The position of current neuron on the image. """
        # 改成multi-neuron
        return [self.get_neurons()["0"][-1], self.get_neurons()["1"][-1]]

    def add_neuron(self, neuron: list):
        print("trigger: Neurons.add_neuron")
        if self.neurons["0"]:
            # 执行匹配算法
            self.match(neuron)
        else:
            assign(self.neurons, neuron)

    def match(self, neuron: list):
        """执行算法（先尝试简单的距离尝试）"""
        print("trigger: Neurons_match")
        previous = self.current_neuron()
        print("previous:", previous)
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
