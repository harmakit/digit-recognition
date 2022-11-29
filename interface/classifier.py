import numpy as np


class ClassifierInterface:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, x_test: np.ndarray) -> list[int]:
        pass
