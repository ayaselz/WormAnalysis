from numpy.core._multiarray_umath import ndarray
import numpy as np

from neurons import Neurons


def distance(coord1, coord2):
    """
    Calculate the distance between two 2-D coordinates

    :param coord1: Numpy array (ndarray)
    :param coord2: Numpy array (ndarray)
    :return:
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def euclidean_distance(group: list):
    # 超出范围怎么报错？
    # 计算保留位数？
    x0, x1, x2 = group
    return np.matrix([distance(x1, x0), distance(x2, x1)])


def angular_differ(group: list):
    x0, x1, x2 = group
    vector1 = x1 - x0
    vector2 = x2 - x1
    division1 = 0.01 if np.linalg.norm(vector1) == 0 else np.linalg.norm(
        vector1)
    division2 = 0.01 if np.linalg.norm(vector2) == 0 else np.linalg.norm(
        vector2)
    # the cos of the angle
    return vector1.dot(vector2) / (division1 * division2)


def quality_score(group1: list, group2: list):
    gamma = 0.01
    theta = 0.1
    # distance cost
    distance_differ1 = euclidean_distance(group1)
    distance_differ2 = euclidean_distance(group2)
    A = np.trace(distance_differ1) - np.trace(distance_differ2)
    distance_cost = abs(np.multiply(np.transpose(A), A))
    # angle cost
    angle_differ1 = angular_differ(group1)
    angle_differ2 = angular_differ(group2)
    angle_cost = abs(angle_differ1 - angle_differ2)
    # result
    return np.exp(-(distance_cost / gamma ** 2) - (angle_cost / theta ** 2))


def coordinate(position: list):
    return np.array([position[0], position[1]])


def position_to_array(neurons: dict | int) -> dict | int:
    if neurons == -1:
        return -1
    if not neurons:
        return {}
    result = {}
    for key in neurons:
        result[key] = coordinate(neurons[key])
    return result


def candidates_to_array(candidates: list) -> list:
    result = []
    for item in candidates:
        result.append(coordinate(item))
    return result


class Assignment(object):
    def __init__(self) -> None:
        self.num = 0  # candidate image num, 随着image num和next last变化
        # list: dict[image_num, neurons[neuron_tag, neuron_physical_coordinate]]
        self.neurons_list: dict[int, dict[str, list]] = {}
        self.previous = {}  # dict
        self.currents = {}  # dict
        self.candidates = []  # list
        # the relationship between the above threes:
        # previous neurons are in (n-1)th image, current neurons are in
        # (n)th image, candidates are in (n+1)th image
        self.amount: int = 1
        self.unit: float = 1
        self.window_radius: float = 6 * self.unit

        self.prediction: dict[str, list] = {}

    def predicted_position(self) -> dict:
        # 当没有previous时，入第二张图
        if self.previous == -1:
            return self.currents
        predicted_position = {}
        for i in range(self.amount):
            key = str(i)
            predicted_position[key] = self.currents[key] + \
                                      (self.currents[key] - self.previous[key])

        return predicted_position

    def classify_candidates(self, radius: float = 0):
        results = {}
        predicted = self.predicted_position()
        for key in predicted:
            results[key] = []
            for item in self.candidates:
                change = predicted[key] - item
                if abs(change[0]) < radius and abs(change[1]) < radius:
                    results[key].append(item)
        self.prediction = results
        # check and modify if under one key the item is empty
        for key in results:
            if not results[key]:
                self.classify_candidates(radius + self.unit)
                break
        # self.prediction = results
        # return results

    # def compare(self):
    #     """考量多种意外的candidate分布：如果某个key下的candidate为空？可能有predicate没有考量到预测错位的情况吗？"""
    #     pass

    def remove_best_matches(self, neuron_key, array_list: list):
        for tag in range(neuron_key + 2, self.amount):
            for item in array_list:
                for compare in self.prediction[str(tag)].copy():
                    if list(compare) != list(item):
                        continue
                    else:
                        after_remove = []
                        for coor in self.prediction[str(tag)]:
                            if list(coor) != list(item):
                                after_remove.append(coor)
                        self.prediction[str(tag)] = after_remove
        print("此时的self.prediction", self.prediction)
                # these schemas cannot be implemented with unknown reason.
                # essentially, cannot compare "compare" and "item" (np?)
                # schema 1
                # lists = self.prediction[str(tag)].copy()
                # for compare in lists:
                #     if (compare != item):
                #         continue
                #     else:
                #         self.prediction[str(tag)].remove(item)
                # schema 2
                # if item in self.prediction[str(tag)]:
                #     self.prediction[str(tag)].remove(item)
                # schema 3
                # if lists.count(item) != 0:
                #     self.prediction[str(tag)].remove(item)

    def best_candidate(self, neuron_key: int):
        print("enter best_candidate")
        # get neurons position in this group
        current_x0 = self.currents[str(neuron_key - 1)]
        current_x1 = self.currents[str(neuron_key)]
        current_x2 = self.currents[str(neuron_key + 1)]
        current_group = [current_x0, current_x1, current_x2]
        # potential results
        # candidates = self.classify_candidates(self.window_radius)
        candidates = self.prediction
        k_range = range(len(candidates[str(neuron_key - 1)]))
        j_range = range(len(candidates[str(neuron_key)]))
        i_range = range(len(candidates[str(neuron_key + 1)]))
        match_result = []
        result_score = None
        for k in k_range:
            for j in j_range:
                for i in i_range:
                    c1 = candidates[str(neuron_key - 1)][k]
                    c2 = candidates[str(neuron_key)][j]
                    c3 = candidates[str(neuron_key + 1)][i]
                    # if c1 == c2 or c2 == c3 or c3 == c1:
                    #     print("发现了重复项")
                    #     continue
                    candidate_group = [c1, c2, c3]
                    score = quality_score(current_group, candidate_group)
                    if result_score is None or score > result_score:
                        match_result = candidate_group
                    # remove the item of best scored in the subsequent combinations
                    self.remove_best_matches(neuron_key, match_result)

        return match_result, result_score

    def results(self) -> dict:
        """储存的position都是物理信息！！！"""
        # 待补充：如果神经元数量小于3怎么处理？
        # 待修正：重叠部分怎么修正？肯定不能直接覆盖
        print("enter results")
        if self.amount == 1:
            return self.result_for_1()
        if self.amount == 2:
            return self.result_for_2()

        self.classify_candidates(self.window_radius)
        print("预测的分类", self.prediction)
        result = {}
        for i in range(1, self.amount - 1):
            # 从ndarray转化成普通list
            assign, _ = self.best_candidate(i)
            print("对", i, "个循环的结果是", assign)
            result[str(i - 1)] = [assign[0][0], assign[0][1]]
            result[str(i)] = [assign[1][0], assign[1][1]]
            result[str(i + 1)] = [assign[2][0], assign[2][1]]

        # 不完备时，没有的坐标设置为-1
        if len(result) < self.amount:
            for i in range(self.amount):
                if str(i) not in result.keys():
                    result[str(i)] = [-1, -1]

        print("最终结果：", result)
        print()
        return result

    def result_for_1(self) -> dict:
        result = {}
        for key in self.currents:
            current_neuron = np.array(self.currents[key])
            candidate = []
            dist = 100000
            for item in self.candidates:
                current_dist = distance(current_neuron, np.array(item))
                if current_dist < dist:
                    dist = current_dist
                    candidate = item
            result[key] = candidate

        # 不完备时，设置-1
        if len(result) < self.amount:
            result["0"] = [-1, -1]

        return result

    def result_for_2(self) -> dict:
        result = {}
        score = None
        length = len(self.candidates)
        if length == 0:
            return {"0": [-1, -1], "1": [-1, -1]}
        if length == 1:
            dist = 100000
            result_key = ''
            for key in self.currents:
                if distance(np.array(self.currents[key]),
                            np.array(self.candidates[0])) < dist:
                    result_key = key
            result[result_key] = self.candidates[0]
            # 完备性
            for i in range(self.amount):
                if str(i) not in result.keys():
                    result[str(i)] = [-1, -1]

            return result

        for j in range(length):
            for i in range(length):
                if i == j:
                    continue
                group = {}
                picked_positions = [self.candidates[i], self.candidates[j]]
                keys = list(self.currents.keys())
                for order in range(len(keys)):
                    group[keys[order]] = picked_positions[order]
                if (score is None) or (self.group_score(group) > score):
                    result = group
                    score = self.group_score(group)

        # 不完备时，没有的坐标设置为-1
        if len(result) < self.amount:
            for i in range(self.amount):
                if str(i) not in result.keys():
                    result[str(i)] = [-1, -1]

        return result

    def group_score(self, group: dict):
        distances = []
        for key in self.currents:
            distances.append(distance(np.array(self.currents[key]),
                                      np.array(group[key])))
        distances = np.array(distances)
        score = - np.average(distances) - np.sqrt(np.var(distances))
        return score

    def assign(self, image_num: int, candidate: list) -> None:
        # 设置现有的previous，current和candidates
        self.num = image_num
        try:
            self.previous = self.neurons_list[image_num - 2]
        except:
            self.previous = -1
        self.currents = self.neurons_list[image_num - 1]
        # 检查-1坐标
        for key in self.currents:
            if self.currents[key][0] == -1:
                self.currents[key] = self.previous[key]
        # 转换成ndarray格式
        self.previous = position_to_array(self.previous)
        self.currents = position_to_array(self.currents)
        self.candidates = candidates_to_array(candidate)
        # 用计算结果赋值
        self.neurons_list[image_num] = self.results()

    def add_neurons(self, image_num: int, neurons: dict) -> None:
        self.neurons_list[image_num] = neurons

    def get_neurons(self, image_num: int) -> dict:
        return self.neurons_list[image_num]

    def swap(self, image_num: int, neuron_num1: str, neuron_num2: str) -> None:
        position1 = self.neurons_list[image_num][neuron_num1]
        position2 = self.neurons_list[image_num][neuron_num2]
        # swap the position data of given tagged neurons in the given image
        self.neurons_list[image_num][neuron_num1] = position2
        self.neurons_list[image_num][neuron_num2] = position1
