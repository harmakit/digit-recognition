import numpy as np


class GuesserInterface:

    def guess(self, unknown_digit: np.ndarray, digits_model: dict[int, np.ndarray]) -> (int, float):
        """guess the digit"""
        pass
