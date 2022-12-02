import numpy as np
from interface.guesser import GuesserInterface


class MeanFeaturesGuesser(GuesserInterface):
    def __init__(self, classifiers: list):
        self.classifiers = classifiers

    def guess(self, unknown_digit_data: np.ndarray) -> (int, float):
        unknown_digit_features = self.get_features(unknown_digit_data)

        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict([unknown_digit_features]))

        guessed_digit = max(predictions, key=predictions.count)[0]
        guessed_digit_confidence = predictions.count(guessed_digit) / len(predictions) * 100

        return guessed_digit, guessed_digit_confidence

    def prepare(self, digits_model: dict[int, np.ndarray]):
        x = []
        y = []

        for digit in digits_model:
            for digit_data in digits_model[digit]:
                x.append(self.get_features(digit_data))
                y.append(digit)

        x = np.array(x)
        y = np.array(y)

        for classifier in self.classifiers:
            classifier.fit(x, y)

    @staticmethod
    def get_features(digit_data: np.ndarray) -> np.ndarray:
        features = []

        window_size = 2
        for i in range(0, 28, window_size):
            for j in range(0, 28, window_size):
                features.append(np.mean(digit_data[i:i + window_size, j:j + window_size]))

        return np.array(features)
