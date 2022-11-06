from interface.guesser import GuesserInterface
from service.data_provider import DataProvider


def run(dp: DataProvider, guesser: GuesserInterface):
    digits_model = dp.get_avg_digits_data(show_plot=True)

    # should be replaced with real digit data >>>
    random_digit_data, real_digit = dp.get_random_digit_data(show_plot=True)
    # <<<

    guessed_digit, guessed_digit_confidence = guesser.guess(random_digit_data, digits_model)

    print(f"Guessed digit: {guessed_digit} with confidence: {guessed_digit_confidence}. Real digit: {real_digit}")
