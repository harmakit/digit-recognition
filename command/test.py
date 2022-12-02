from interface.guesser import GuesserInterface
from service.data_provider import DataProvider

SAMPLE_RATE = 0.1


def test(dp: DataProvider, guesser: GuesserInterface):
    train_digits_model = {}
    test_digits_model = {}

    if guesser.__class__.__name__ == "DiffGuesser":
        train_digits_model = dp.get_compiled_digits_models_data()
        test_digits_model = dp.get_digits_models_data()
    else:
        digits_model = dp.get_digits_models_data()
        for digit in digits_model:
            test_digits_model[digit] = digits_model[digit][:int(len(digits_model[digit]) * SAMPLE_RATE)]
            train_digits_model[digit] = digits_model[digit][int(len(digits_model[digit]) * SAMPLE_RATE):]

    guesser.prepare(train_digits_model)

    tests_count = 0
    correct_guesses = 0
    for digit in test_digits_model:
        for digit_data in test_digits_model[digit]:
            tests_count += 1
            guess, _ = guesser.guess(digit_data)
            if guess == digit:
                correct_guesses += 1

    print(
        f"Guessed {correct_guesses} out of {tests_count} digits correctly. Accuracy: {correct_guesses / tests_count * 100}%")
    pass
