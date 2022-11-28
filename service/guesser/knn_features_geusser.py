import numpy as np
from interface.guesser import GuesserInterface


class KnnFeaturesGeusser(GuesserInterface):
    def __init__(self):
        self.models = {}

    def guess(self, unknown_digit_data: np.ndarray) -> (int, float):
        # calculate the euclidien distance
        diff = {}
        for digit, avg_digit_data in self.models.items():
            diff[digit] = np.sqrt(np.sum((avg_digit_data - unknown_digit_data)**2))
            diff[digit].sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(4):
            neighbors.append(diff[i][0])
        output_values = [row[-1] for row in neighbors]
        guessed_digit = max(set(output_values), key=output_values.count)

        # get the confidence of the guess
        similarity = 1 - (diff[guessed_digit] / (np.size(unknown_digit_data) * 255))
        guessed_digit_confidence = similarity * 100
        
        return guessed_digit, guessed_digit_confidence
   
    
    def prepare(self, digits_model: dict[int, np.ndarray]):
        self.models = digits_model
	    
    

