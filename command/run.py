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

    if guesser.__class__.__name__ == "DiffGuesser":
        digits_model = dp.get_compiled_digits_models_data()
    else:
        digits_model = dp.get_digits_models_data()
    guesser.prepare(digits_model)

    image_processor = ImageProcessor()
    image_data = image_processor.get_image(file_path)

    guessed_digit, guessed_digit_confidence = guesser.guess(image_data)
    guessed_digit_confidence = round(guessed_digit_confidence, 2)

    print(f"Guessed digit: {guessed_digit} with confidence: {guessed_digit_confidence}%")
