import numpy as np
from interface.guesser import GuesserInterface


class MeanFeaturesGuesser(GuesserInterface):
    def __init__(self, classifiers: list):
        self.classifiers = classifiers
        self.models = {}

    def guess(self, unknown_digit_data: np.ndarray, digits_model: dict[int, np.ndarray]) -> (int, float):
        unknown_digit_features = self.get_features(unknown_digit_data)

        predictions = []
        for model in self.models:
            predictions.append(self.models[model].predict([unknown_digit_features]))

        guessed_digit = max(predictions, key=predictions.count)
        guessed_digit_confidence = predictions.count(guessed_digit) / len(predictions) * 100

        return guessed_digit, guessed_digit_confidence

    def prepare(self, digits_model: dict[int, np.ndarray]):
        models = {}
        features = {}

        for digit in digits_model:
            features[digit] = self.get_features(digits_model[digit])
        features = np.array(list(features.values()))

        for classifier in self.classifiers:
            models[classifier.__class__.__name__] = classifier.fit(features, list(digits_model.keys()))

        self.models = models

    @staticmethod
    def get_features(digit_data: np.ndarray) -> np.ndarray:
        features = []

        window_size = 2
        for i in range(0, 28, window_size):
            for j in range(0, 28, window_size):
                features.append(np.mean(digit_data[i:i + window_size, j:j + window_size]))

        return np.array(features)
