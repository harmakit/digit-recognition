import numpy as np
from interface.guesser import GuesserInterface


class DiffGuesser(GuesserInterface):

    def __init__(self):
        self.models = {}

    def guess(self, unknown_digit_data: np.ndarray) -> (int, float):
        # get the difference between the unknown digit and the average digit data for each digit in the model
        diff = {}
        for digit, avg_digit_data in self.models.items():
            diff[digit] = np.sum(np.abs(avg_digit_data - unknown_digit_data))

        # get the digit with the lowest difference
        guessed_digit = min(diff, key=diff.get)

        # get the confidence of the guess
        similarity = 1 - (diff[guessed_digit] / (np.size(unknown_digit_data) * 255))
        guessed_digit_confidence = similarity * 100

        return guessed_digit, guessed_digit_confidence

    def prepare(self, digits_model: dict[int, np.ndarray]):
        self.models = digits_model
