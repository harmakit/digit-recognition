from simple_term_menu import TerminalMenu
from command.run import run
from command.test import test
from command.update import update
from service.data_provider import DataProvider, DataProviderSource, DataProviderAlgorithm
from service.guesser_factory import GuesserFactory, GuesserType

actions = ["[r] run recognizer", "[t] test recognizer", "[u] update data from images", "[e] exit"]
actions_menu = TerminalMenu(actions, title="Choose action")

data_sources = ["[d] data dir", "[m] mnist dataset"]
data_sources_menu = TerminalMenu(data_sources, title="Choose data source")

data_algorithms = ["[a] average", "[c] cumulative"]
data_algorithms_menu = TerminalMenu(data_algorithms, title="Choose data provider algorithm")

guesser_algorithms = ["[a] run all algorithms", "[d] diff - comparing by pixel value",
                      "[m] mean – mean in window feature", "[h] haar features"]
guesser_algorithms_menu = TerminalMenu(guesser_algorithms, title="Choose guesser algorithm")

while True:
    menu_entry_index = actions_menu.show()

    if menu_entry_index == 3:
        exit()

    data_provider_source = None
    data_provider_algorithm = None

    guesser_factory = GuesserFactory()
    guessers = []

    if menu_entry_index == 0 or menu_entry_index == 1:
        data_source_index = data_sources_menu.show()
        if data_source_index == 0:
            data_provider_source = DataProviderSource.DATADIR
        elif data_source_index == 1:
            data_provider_source = DataProviderSource.MNIST
        else:
            exit('Invalid data source')

        guesser_algorithm_index = guesser_algorithms_menu.show()
        if guesser_algorithm_index == 0:
            guessers.append(guesser_factory.get_guesser(GuesserType.DIFF))
            guessers.append(guesser_factory.get_guesser(GuesserType.MEAN))
            guessers.append(guesser_factory.get_guesser(GuesserType.HAAR))
        elif guesser_algorithm_index == 1:
            guessers.append(guesser_factory.get_guesser(GuesserType.DIFF))
        elif guesser_algorithm_index == 2:
            guessers.append(guesser_factory.get_guesser(GuesserType.MEAN))
        elif guesser_algorithm_index == 3:
            guessers.append(guesser_factory.get_guesser(GuesserType.HAAR))
        else:
            exit('Invalid guesser algorithm')

        data_provider_algorithm = None
        if guesser_algorithm_index == 0 or guesser_algorithm_index == 1:
            data_algorithm_index = data_algorithms_menu.show()
            if data_algorithm_index == 0:
                data_provider_algorithm = DataProviderAlgorithm.AVG
            elif data_algorithm_index == 1:
                data_provider_algorithm = DataProviderAlgorithm.CUMULATIVE

    if menu_entry_index == 2:
        data_provider_source = DataProviderSource.DATADIR

    dp = DataProvider(data_provider_source, data_provider_algorithm)

    if menu_entry_index == 0:  # run
        for guesser in guessers:
            print(guesser.__class__.__name__ + ':')
            run(dp, guesser)
            print('')
    elif menu_entry_index == 1:  # test
        for guesser in guessers:
            print(guesser.__class__.__name__ + ':')
            run(dp, guesser)
            print('')
    elif menu_entry_index == 2:  # update
        update()
