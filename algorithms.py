"""
Assignment class is implemented with tracking algorithms. Functions are used to
support algorithms in Assignment. The input port is
Assignment.add_neuron(image_num: int, candidates: list[ndarray]). Candidates
are recognized highlights. The output port is get_neurons(image_num: int), with
return type dict[str, ndarray].

分配类是用跟踪算法实现的。 函数用于辅助计算Assignment中的追踪算法。 输入端口是
assignment.add_neuron(image_num: int, Candidates: list[ndarray])。 Candidates
是已经识别出的亮点。输出端口是get_neurons(image_num: int)，其返回值的类型是
dict[str, ndarray]。
"""

from numpy.core._multiarray_umath import ndarray
import numpy as np

from neurons import Neurons


def distance(coord1: ndarray, coord2: ndarray):
    """
    Calculate the distance between two 2-D coordinates

    :param coord1: Numpy array (ndarray)
    :param coord2: Numpy array (ndarray)
    :return:  distance between two coordinates
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def euclidean_distance(group: list):
    """
    Calculate two side length of the triangle formed by the
    given three vertexes.
    :param group: a list of coordinates of three points.
    :return: an array containing with two side length.
    """
    x0, x1, x2 = group
    return np.matrix([distance(x1, x0), distance(x2, x1)])


def angular_differ(group: list):
    """
    Calculate the angle formed by the given three vertexes.
    The middle point works as the vertex.
    :param group: a list of coordinates of three points.
    :return: the cos of the angle
    """
    x0, x1, x2 = group
    vector1 = x1 - x0
    vector2 = x2 - x1
    division1 = 0.01 if np.linalg.norm(vector1) == 0 else np.linalg.norm(
        vector1)
    division2 = 0.01 if np.linalg.norm(vector2) == 0 else np.linalg.norm(
        vector2)
    # the cos of the angle
    return vector1.dot(vector2) / (division1 * division2)


def quality_score(group1: list, group2: list) -> float:
    """
    The cost function that can determine the similarity score of two groups of
    points. The similarity increases with the reduction of distance or angular
    differences.
    :param group1: a list of coordinates of three points.
    :param group2: the other list of coordinates of three points.
    :return:
    """
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


def coordinate(position: list) -> ndarray:
    """
    Transform a list to array format.
    :param position: a list representing position
    :return:
    """
    return np.array([position[0], position[1]])


def position_to_array(neurons: dict | int) -> dict | int:
    """
    Transform a dictionary of positions to array format. The value -1 represents
    the vacancies.
    :param neurons: dict[image_num: str, coordinate: list]
    :return: reformed position dictionary
    """
    if neurons == -1:
        return -1
    if not neurons:
        return {}
    result = {}
    for key in neurons:
        result[key] = coordinate(neurons[key])
    return result


def candidates_to_array(candidates: list) -> list:
    """
    Transform a list of positions to array format.
    :param candidates: a list of positions
    :return: reformed position list
    """
    result = []
    for item in candidates:
        result.append(coordinate(item))
    return result


class Assignment(object):
    """
    The tracking algorithm implementation.
    """
    def __init__(self) -> None:
        self.num = 0  # candidate image num at t+1
        self.neurons_list: dict[int, dict[str, list]] = {}
        # representing image at t-1, will be read from neurons_list
        self.previous: dict = {}
        # representing image at t, will be read from neurons_list
        self.currents: dict = {}
        self.candidates: list = []
        # the relationship between the above threes:
        # previous neurons are in (n-1)th image, current neurons are in
        # (n)th image, candidates are in (n+1)th image
        self.amount: int = 1
        # the unit of distance, 1 without physical position files or other value
        # assigned with position file.
        self.unit: float = 1
        self.window_radius: float = 6 * self.unit

    def predicted_position(self) -> dict:
        """
        Using the t-1 and t image to predict the centers of possible ranges for
        neurons in image t+1.
        :return: the centers of search windows
        """
        # when there is no previous image, import the current one
        if self.previous == -1:
            return self.currents
        predicted_position = {}
        for i in range(self.amount):
            key = str(i)
            predicted_position[key] = self.currents[key] + \
                                      (self.currents[key] - self.previous[key])

        return predicted_position

    def classify_candidates(self, radius: float = 0) -> dict[str, list]:
        """
        Allocating the recognized highlights to each search window.
        Search windows are calculated here rather than in center predication.
        :param radius: the size of each window
        :return: a dictionary of allocation results
        """
        results = {}
        predicted = self.predicted_position()
        for key in predicted:
            results[key] = []
            for item in self.candidates:
                change = predicted[key] - item
                if change[0] < radius \
                        and change[1] < radius:
                    results[key].append(item)
        # check and modify if under one key the item is empty
        for key in results:
            if not results[key]:
                results = self.classify_candidates(radius + self.unit)
        return results

    def best_candidate(self, neuron_key: int):
        """
        Calculate the similarity of all the possible combinations and return the
        result of assignment.
        :param neuron_key:
        :return:
        """
        # get neurons position in this group
        current_x0 = self.currents[str(neuron_key - 1)]
        current_x1 = self.currents[str(neuron_key)]
        current_x2 = self.currents[str(neuron_key + 1)]
        current_group = [current_x0, current_x1, current_x2]
        # potential results
        candidates = self.classify_candidates(self.window_radius)
        k_range = range(len(candidates[str(neuron_key - 1)]))
        j_range = range(len(candidates[str(neuron_key)]))
        i_range = range(len(candidates[str(neuron_key + 1)]))
        match_result = []
        result_score = None
        for k in k_range:
            for j in j_range:
                for i in i_range:
                    candidate_group = [candidates[str(neuron_key - 1)][k],
                                       candidates[str(neuron_key)][j],
                                       candidates[str(neuron_key + 1)][i]]
                    score = quality_score(current_group, candidate_group)
                    if result_score is None or score > result_score:
                        match_result = candidate_group
        return match_result, result_score

    def results(self) -> dict:
        """
        The final result after assigning all the neurons.
        :return: dict[neuron_num: str, coordinate: ndarray]
        """
        if self.amount == 1:
            return self.result_for_1()
        if self.amount == 2:
            return self.result_for_2()

        result = {}
        for i in range(1, self.amount - 1):
            # transform ndarray to list
            # 从ndarray转化成普通list
            assign, _ = self.best_candidate(i)
            result[str(i - 1)] = [assign[0][0], assign[0][1]]
            result[str(i)] = [assign[1][0], assign[1][1]]
            result[str(i + 1)] = [assign[2][0], assign[2][1]]

        # with incompleteness, the coordinate is set as -1
        # 不完备时，没有的坐标设置为-1
        if len(result) < self.amount:
            for i in range(self.amount):
                if str(i) not in result.keys():
                    result[str(i)] = [-1, -1]

        return result

    def result_for_1(self) -> dict:
        """
        Process the assignment when the number of neurons is 1
        :return: dict[neuron_num: str, coordinate: ndarray]
        """
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

        # with incompleteness, the coordinate is set as -1
        # 不完备时，没有的坐标设置为-1
        if len(result) < self.amount:
            result["0"] = [-1, -1]

        return result

    def result_for_2(self) -> dict:
        """
        Process the assignment when the number of neurons is 1
        :return: dict[neuron_num: str, coordinate: ndarray]
        """
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
            # completeness
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

        # with incompleteness, the coordinate is set as -1
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
        """
        The input port of this class. With being triggered, the process of
        assignment/tracking is executed automatically.
        :param image_num: the number of the calculated image
        :param candidate: the positions of recognized highlights in image
        """
        self.num = image_num
        try:
            self.previous = self.neurons_list[image_num - 2]
        except:
            self.previous = -1
        self.currents = self.neurons_list[image_num - 1]
        # check whether existing -1 coordinates
        # 检查-1坐标
        for key in self.currents:
            if self.currents[key][0] == -1:
                self.currents[key] = self.previous[key]
        # transform to ndarray
        # 转换成ndarray格式
        self.previous = position_to_array(self.previous)
        self.currents = position_to_array(self.currents)
        self.candidates = candidates_to_array(candidate)
        # add the result in neurons_list
        # 将计算结果添加进neurons_list
        self.neurons_list[image_num] = self.results()

    def add_neurons(self, image_num: int, neurons: dict) -> None:
        """
        Similar to assign(). It works for the initial image.
        :param image_num: the number of the calculated image
        :param neurons: dict[neuron_num: str, coordinate: ndarray],
        the positions of recognized highlights in image
        """
        self.neurons_list[image_num] = neurons

    def get_neurons(self, image_num: int) -> dict:
        """
        The input port of this class. Returns the calculated results of the
        specified image.
        :param image_num: the number of the specified image
        :return: dict[neuron_num: str, coordinate: ndarray],
        the neurons positions in the specified image.
        """
        return self.neurons_list[image_num]

    def swap(self, image_num: int, neuron_num1: str, neuron_num2: str) -> None:
        """
        Exchange the information in neurons_list of the specified neurons
        in the specified image.
        :param image_num: the number of the specified image
        :param neuron_num1: the number of the specified neuron
        :param neuron_num2: the other number of the specified neuron
        """
        position1 = self.neurons_list[image_num][neuron_num1]
        position2 = self.neurons_list[image_num][neuron_num2]
        # swap the position data of given tagged neurons in the given image
        self.neurons_list[image_num][neuron_num1] = position2
        self.neurons_list[image_num][neuron_num2] = position1
