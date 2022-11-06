import os
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras


class DataProviderSource(Enum):
    MNIST = 0
    DATADIR = 1


DATADIR = os.path.dirname(os.path.abspath(__file__)) + "/../data"


class DataProvider:
    digits_data: list[list[int, np.ndarray]]
    avg_digits_data: dict[int, np.ndarray]

    def __init__(self, source: DataProviderSource):
        switcher = {
            DataProviderSource.MNIST: self.__load_mnist,
            DataProviderSource.DATADIR: self.__load_datadir
        }
        self.digits_data = switcher.get(source)()
        self.avg_digits_data = self.__init_avg_digits_data()

    @staticmethod
    def __load_mnist() -> list[list[int, np.ndarray]]:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        digits_data = []
        for i in range(len(x_test)):
            digit = y_test[i]
            digits_data.append([digit, x_test[i]])
        return digits_data

    @staticmethod
    def __load_datadir() -> list[list[int, np.ndarray]]:
        digits_data = []
        for i in range(10):
            digits_array_data = np.load(f'{DATADIR}/npy/digit_{i}.npy')
            for digit_data in digits_array_data:
                digits_data.append([i, digit_data])
        return digits_data

    def __init_avg_digits_data(self) -> dict[int, np.ndarray]:
        avg_digit_data = {}
        for i in range(len(self.digits_data)):
            digit = self.digits_data[i][0]
            current_avg_digit_data = avg_digit_data.get(digit, self.digits_data[i][1])
            avg_digit_data[digit] = (current_avg_digit_data + self.digits_data[i][1]) / 2

        return dict(sorted(avg_digit_data.items()))

    def get_avg_digits_data(self, show_plot=False) -> dict[int, np.ndarray]:
        if show_plot:
            # multiple plot the average digit data
            fig, axs = plt.subplots(2, 5)
            fig.suptitle('Average Digit Data')
            for i in range(2):
                for j in range(5):
                    digit = i * 5 + j
                    axs[i, j].imshow(self.avg_digits_data[digit], cmap='gray')
            plt.show()
        return self.avg_digits_data

    def get_random_digit_data(self, show_plot=False) -> (np.ndarray, int):
        random_index = np.random.randint(0, len(self.digits_data))
        random_digit = self.digits_data[random_index][0]
        random_digit_data = self.digits_data[random_index][1]

        if show_plot:
            title = f'Random Digit Data: {random_digit}'
            plt.title(title)
            plt.imshow(random_digit_data, cmap='gray')
            plt.show()

        return random_digit_data, random_digit
