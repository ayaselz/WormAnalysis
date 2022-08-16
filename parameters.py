__author__ = "{{Wencan Peng}} ({{46222378}})"
__email__ = "wencan.peng@uqconnect.edu.au"
__date__ = "16/08/2022"
__version__ = "2.0"

class Parameters(object):
    def __init__(self, alpha: int = 3, beta: int = 0,
                 peak_circle: int = 6, peak_ratio: float = 0.4,
                 row_bias: int = 0, column_bias: int = 0,
                 label_radius: int = 7,
                 right_black: int = 0, left_black: int = 0,
                 right_black_bias: int = 0, left_black_bias: int = 0,
                 right_circle: int = 5, right_ratio: float = 0.6,
                 left_circle: int = 5, left_ratio: float = 0.6) -> None:
        self.__alpha = alpha
        self.__beta = beta
        self.__peak_circle = peak_circle
        self.__peak_ratio = peak_ratio
        self.__row_bias = row_bias
        self.__column_bias = column_bias
        self.__label_radius = label_radius
        self.__right_black = right_black
        self.__left_black = left_black
        self.__right_black_bias = right_black_bias
        self.__left_black_bias = left_black_bias
        self.__right_circle = right_circle
        self.__right_ratio = right_ratio
        self.__left_circle = left_circle
        self.__left_ratio = left_ratio

    @property
    def alpha(self) -> int:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: int) -> None:
        self.__alpha = alpha

    @property
    def beta(self) -> int:
        return self.__beta

    @beta.setter
    def beta(self, beta: int) -> None:
        self.__beta = beta

    @property
    def peak_circle(self) -> int:
        return self.__peak_circle

    @peak_circle.setter
    def peak_circle(self, peak_circle: int) -> None:
        self.__peak_circle = peak_circle

    @property
    def peak_ratio(self) -> float:
        return self.__peak_ratio

    @peak_ratio.setter
    def peak_ratio(self, peak_ratio: float) -> None:
        self.__peak_ratio = peak_ratio

    # row_bias: int = 0, column_bias: int = 0,
    @property
    def row_bias(self) -> int:
        return self.__row_bias

    @row_bias.setter
    def row_bias(self, row_bias: int) -> None:
        self.__row_bias = row_bias

    @property
    def column_bias(self) -> int:
        return self.__column_bias

    @column_bias.setter
    def column_bias(self, column_bias: int) -> None:
        self.__column_bias = column_bias

# label_radius: int = 7,
    @property
    def label_radius(self) -> int:
        return self.__label_radius

    @label_radius.setter
    def label_radius(self, label_radius: int) -> None:
        self.__label_radius = label_radius
# right_black: int = 0, left_black: int = 0,
    @property
    def right_black(self) -> int:
        return self.__right_black

    @right_black.setter
    def right_black(self, right_black: int) -> None:
        self.__right_black = right_black

    @property
    def left_black(self) -> int:
        return self.__left_black

    @left_black.setter
    def left_black(self, left_black: int) -> None:
        self.__left_black = left_black
# right_black_bias: int = 0, left_black_bias: int = 0,
    @property
    def right_black_bias(self) -> int:
        return self.__right_black_bias

    @right_black_bias.setter
    def right_black_bias(self, right_black_bias: int) -> None:
        self.__right_black_bias = right_black_bias

    @property
    def left_black_bias(self) -> int:
        return self.__left_black_bias

    @left_black_bias.setter
    def left_black_bias(self, left_black_bias: int) -> None:
        self.__left_black_bias = left_black_bias
# right_circle: int = 5, right_ratio: float = 0.6,
    @property
    def right_circle(self) -> int:
        return self.__right_circle

    @right_circle.setter
    def right_circle(self, right_circle: int) -> None:
        self.__right_circle = right_circle

    @property
    def right_ratio(self) -> float:
        return self.__right_ratio

    @right_ratio.setter
    def right_ratio(self, right_ratio: float) -> None:
        self.__right_ratio = right_ratio
# left_circle: int = 5, left_ratio: float = 0.6
    @property
    def left_circle(self) -> int:
        return self.__left_circle

    @left_circle.setter
    def left_circle(self, left_circle: int) -> None:
        self.__left_circle = left_circle

    @property
    def left_ratio(self) -> float:
        return self.__left_ratio

    @left_ratio.setter
    def left_ratio(self, left_ratio: float) -> None:
        self.__left_ratio = left_ratio

