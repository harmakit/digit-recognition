import os
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST


class DataProviderSource(Enum):
    MNIST = 0
    DATADIR = 1


class DataProviderAlgorithm(Enum):
    AVG = 0
    CUMULATIVE = 1


DATADIR = os.path.dirname(os.path.abspath(__file__)) + "/../data"


class DataProvider:
    digits_data: list[list[int, np.ndarray]]
    compiled_digits_data: dict[int, np.ndarray] | None = None

    def __init__(self, source: DataProviderSource, algorithm: DataProviderAlgorithm | None):
        switcher = {
            DataProviderSource.MNIST: self.__load_mnist,
            DataProviderSource.DATADIR: self.__load_datadir
        }
        self.digits_data = switcher.get(source)()

        if algorithm is not None:
            switcher = {
                DataProviderAlgorithm.AVG: self.__init_avg_digits_data,
                DataProviderAlgorithm.CUMULATIVE: self.__init_cumulative_digits_data
            }
            self.compiled_digits_data = switcher.get(algorithm)()

    @staticmethod
    def __load_mnist() -> list[list[int, np.ndarray]]:
        mnist_data = MNIST(f'{DATADIR}/mnist')
        images, labels = mnist_data.load_testing()

        digits_data = []
        for i in range(len(images)):
            digit = labels[i]
            # convert to numpy 28x28 array
            digit_data = np.array(images[i]).reshape((28, 28))
            digits_data.append([digit, digit_data])
        return digits_data

    @staticmethod
    def __load_datadir() -> list[list[int, np.ndarray]]:
        digits_data = []
        for i in range(10):
            digits_array_data = np.load(f'{DATADIR}/npy/digit_{i}.npy')
            for digit_data in digits_array_data:
                digits_data.append([i, digit_data])
        return digits_data

    # cumulative method
    def __init_cumulative_digits_data(self) -> dict[int, np.ndarray]:
        cumulative_digit_data = {}
        for i in range(len(self.digits_data)):
            digit = self.digits_data[i][0]
            current_avg_digit_data = cumulative_digit_data.get(digit, np.zeros((28, 28)))
            cumulative_digit_data[digit] = (current_avg_digit_data + self.digits_data[i][1]) / 2

        return dict(sorted(cumulative_digit_data.items()))

    # average method
    def __init_avg_digits_data(self) -> dict[int, np.ndarray]:
        avg_digit_data = {}
        lengths = [0] * 10
        for i in range(len(self.digits_data)):
            digit = self.digits_data[i][0]
            current_avg_digit_data = avg_digit_data.get(digit, np.zeros((28, 28)))
            avg_digit_data[digit] = current_avg_digit_data + self.digits_data[i][1]
            lengths[digit] += 1

        for i in range(10):
            # calc average
            avg_digit_data[i] = avg_digit_data[i] / lengths[i]
            # convert to uint8
            avg_digit_data[i] = avg_digit_data[i].astype(np.uint8)

        return dict(sorted(avg_digit_data.items()))

    def get_digits_models_data(self) -> dict[int, np.ndarray]:
        digits_data = {}

        for i in range(len(self.digits_data)):
            digit = self.digits_data[i][0]
            digit_data = self.digits_data[i][1]

            if digit not in digits_data:
                digits_data[digit] = []
            digits_data[digit].append(digit_data)

        return dict(sorted(digits_data.items()))

    def get_compiled_digits_models_data(self, show_plot=False) -> dict[int, np.ndarray]:
        if self.compiled_digits_data is None:
            raise Exception('No compiled digits data')

        if show_plot:
            # multiple plot the average digit data
            fig, axs = plt.subplots(2, 5)
            fig.suptitle('Compiled Digit Data')
            for i in range(2):
                for j in range(5):
                    digit = i * 5 + j
                    axs[i, j].imshow(self.compiled_digits_data[digit], cmap='gray')
            plt.show()
        return self.compiled_digits_data
