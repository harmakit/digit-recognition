import os
import sys

# suppress rebuild TensorFlow warnings >>>
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# <<<

from simple_term_menu import TerminalMenu
from command.run import run
from command.test import test
from command.update import update
from service.data_provider import DataProvider, DataProviderSource
from service.diff_guesser import DiffGuesser

actions = ["[r] run recognizer", "[t] test recognizer", "[u] update data from images"]
terminal_menu = TerminalMenu(actions, title="Choose action")
menu_entry_index = terminal_menu.show()

data_provider_source = DataProviderSource.MNIST
if menu_entry_index == 0 or menu_entry_index == 1:
    data_sources = ["[d] data dir", "[m] mnist dataset"]
    terminal_menu = TerminalMenu(data_sources, title="Choose data source")
    data_source_index = terminal_menu.show()
    if data_source_index == 0:
        data_provider_source = DataProviderSource.DATADIR
    elif data_source_index == 1:
        data_provider_source = DataProviderSource.MNIST

dp = DataProvider(data_provider_source)
guesser = DiffGuesser()

if menu_entry_index == 0:  # run
    run(dp, guesser)
elif menu_entry_index == 1:  # test
    test(dp, guesser)
elif menu_entry_index == 2:  # update
    update()