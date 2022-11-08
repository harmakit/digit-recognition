import os
from simple_term_menu import TerminalMenu
from command.run import run
from command.test import test
from command.update import update
from service.data_provider import DataProvider, DataProviderSource, DataProviderAlgorithm
from service.diff_guesser import DiffGuesser

actions = ["[r] run recognizer", "[t] test recognizer", "[u] update data from images"]
actions_menu = TerminalMenu(actions, title="Choose action")

data_sources = ["[d] data dir", "[m] mnist dataset"]
data_sources_menu = TerminalMenu(data_sources, title="Choose data source")

data_algorithms = ["[c] cumulative", "[a] average"]
data_algorithms_menu = TerminalMenu(data_algorithms, title="Choose data provider algorithm")

menu_entry_index = actions_menu.show()

data_provider_source = DataProviderSource.MNIST
data_provider_algorithm = DataProviderAlgorithm.CUMULATIVE

if menu_entry_index == 0 or menu_entry_index == 1:
    data_source_index = data_sources_menu.show()
    if data_source_index == 0:
        data_provider_source = DataProviderSource.DATADIR
    elif data_source_index == 1:
        data_provider_source = DataProviderSource.MNIST

    data_algorithm_index = data_algorithms_menu.show()
    if data_algorithm_index == 0:
        data_provider_algorithm = DataProviderAlgorithm.CUMULATIVE
    elif data_algorithm_index == 1:
        data_provider_algorithm = DataProviderAlgorithm.AVG

dp = DataProvider(data_provider_source, data_provider_algorithm)
guesser = DiffGuesser()

if menu_entry_index == 0:  # run
    run(dp, guesser)
elif menu_entry_index == 1:  # test
    test(dp, guesser)
elif menu_entry_index == 2:  # update
    update()
