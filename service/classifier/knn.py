from interface.classifier import ClassifierInterface
from enum import Enum
import numpy as np


class DistMetric(Enum):
    EUCLIDEAN = 0


def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class KNeighborsClassifier(ClassifierInterface):
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
        self.x_train = None
        self.y_train = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: np.ndarray) -> list[int]:
        neighbors = []
        for x in x_test:
            distances = self.dist_metric(x, self.x_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
