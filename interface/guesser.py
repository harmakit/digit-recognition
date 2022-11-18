import numpy as np


class GuesserInterface:

    def guess(self, unknown_digit: np.ndarray) -> (int, float):
        """guess the digit"""
        pass

    def prepare(self, digits_model: dict[int, np.ndarray]):
        """prepare the guesser if needed"""
        pass
