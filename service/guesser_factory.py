from enum import Enum
from service.classifier.knn import KNeighborsClassifier
from service.guesser.diff_guesser import DiffGuesser
from service.guesser.haar_features_guesser import HaarFeaturesGuesser
from service.guesser.mean_features_guesser import MeanFeaturesGuesser


class GuesserType(Enum):
    DIFF = 0
    MEAN = 1
    HAAR = 2


class GuesserFactory:
    feature_methods_classifiers = [
        KNeighborsClassifier(),
    ]

    def get_guesser(self, guesser_type: GuesserType):
        switcher = {
            GuesserType.DIFF: self.__get_diff_guesser,
            GuesserType.MEAN: self.__get_mean_features_guesser,
            GuesserType.HAAR: self.__get_haar_features_guesser
        }
        return switcher.get(guesser_type)()

    @staticmethod
    def __get_diff_guesser():
        return DiffGuesser()

    def __get_mean_features_guesser(self):
        return MeanFeaturesGuesser(self.feature_methods_classifiers)

    def __get_haar_features_guesser(self):
        return HaarFeaturesGuesser(self.feature_methods_classifiers)
