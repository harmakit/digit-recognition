import os

from interface.guesser import GuesserInterface
from service.data_provider import DataProvider
from service.image_processor import ImageProcessor


def run(dp: DataProvider, guesser: GuesserInterface):
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/../digit.png"
    # check if digit.png exists
    if not os.path.isfile(file_path):
        print("File digit.png not found. Please place it in the root directory of the project.")
        return

    digits_model = dp.get_digits_models_data(show_plot=True)

    image_processor = ImageProcessor()
    image_data = image_processor.get_image(file_path)

    guessed_digit, guessed_digit_confidence = guesser.guess(image_data, digits_model)

    print(f"Guessed digit: {guessed_digit} with confidence: {guessed_digit_confidence}")
