from interface.guesser import GuesserInterface
from service.data_provider import DataProvider


def test(dp: DataProvider, guesser: GuesserInterface):
    # test the guesser with random mnist digits
    digits_model = dp.get_digits_models_data()
    tests_len = 1000
    correct_guesses = 0
    for i in range(tests_len):
        random_digit_data, random_digit = dp.get_random_digit_data()
        guessed_digit, guessed_digit_confidence = guesser.guess(random_digit_data, digits_model)
        if guessed_digit == random_digit:
            correct_guesses += 1

    print(
        f"Guessed {correct_guesses} out of {tests_len} digits correctly. Accuracy: {correct_guesses / tests_len * 100}%")
    pass
