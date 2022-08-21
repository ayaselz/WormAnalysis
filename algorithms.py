import numpy as np

from neurons import Neurons


def distance(coord1, coord2):
    """ Calculate the distance betwwen two 2-D coordinates"""
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
    return np.array(position[0], position[1])


def position_to_array(neurons: Neurons) -> dict:
    if not neurons.assigned:
        return {}
    result = {}
    for key in neurons.assigned:
        result[key] = coordinate(neurons.assigned[key])
    return result


def candidates_to_array(candidates: list) -> list:
    result = []
    for item in candidates:
        result.append(coordinate(item))
    return result


class Assignment(object):
    def __init__(self, amount: int, candidates: Neurons,
                 current_neurons: Neurons | int,
                 previous_neurons: Neurons | int) -> None:
        self.previous = position_to_array(previous_neurons)  # dict
        self.currents = position_to_array(current_neurons)  # dict
        self.candidates = candidates_to_array(candidates.potential)  # list
        # the relationship between the above threes:
        # previous neurons are in (n-1)th image, current neurons are in
        # (n)th image, candidates are in (n+1)th image
        self.amount = amount
        trans_ratio = candidates.header * (1200 / 512)
        self.window_radius: float = 6 * trans_ratio

    def predicted_position(self) -> dict:
        predicted_position = {}
        for i in range(self.amount):
            key = str(i)
            predicted_position[key] = self.currents[key] + \
                                      (self.currents[key] - self.previous[key])
        return predicted_position

    def classify_candidates(self) -> dict[str, list]:
        results = {}
        predicted = self.predicted_position()
        for key in predicted:
            results[key] = []
            for item in self.candidates:
                change = predicted[key] - item
                if change[0] < self.window_radius \
                        and change[1] < self.window_radius:
                    results[key].append(item)
        return results

    def compare(self):
        """考量多种意外的candidate分布：如果某个key下的candidate为空？可能有predicate没有考量到预测错位的情况吗？"""
        pass

    def best_candidate(self, neuron_key: int):
        # get neurons position in this group
        current_x0 = self.currents[str(neuron_key - 1)]
        current_x1 = self.currents[str(neuron_key)]
        current_x2 = self.currents[str(neuron_key + 1)]
        current_group = [current_x0, current_x1, current_x2]
        # potential results
        candidates = self.classify_candidates()
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
        """储存的position都是物理信息！！！"""
        # 待补充：如果神经元数量小于3怎么处理？
        # 待修正：重叠部分怎么修正？肯定不能直接覆盖
        result = {}
        for i in range(1, self.amount - 1):
            # 从ndarray转化成普通list
            assign, _ = self.best_candidate(i)
            result[str(i - 1)] = [assign[0][0], assign[0][1]]
            result[str(i)] = [assign[1][0], assign[1][1]]
            result[str(i + 1)] = [assign[2][0], assign[2][1]]
        return result
