"""
In this script we show 3 ways to run a CLASSIFICATION training session. We will use the provided miniMNIST to do that.

Note about classification training: We generally observed that all decoders learn this small miniMNISt apart from the last decoder, which accuracy stays at ~20%. We will maybe observe the same thing with bigger dataset, and the reason could be that the last decoder only has 2048 weights (against ~100k to ~800k of the other decoders), the rest of the net is frozen, so it might be too difficult to adapt those few weights to a new task. If that's the case, we might just exclude the last decoder.
"""
import os
import toml
from src.utils.decoder.train import run_train

##
# Firstly, we can use just few training options. The options in "default_train_config" will be used for all other options.
data_folder = 'data/examples/miniMNIST'
run_train(training={'train_dataset': f'{data_folder}/training',
                         'test_datasets': [f'{data_folder}/testing1',
                                           f'{data_folder}/testing2']},
          saving_folders={'result_folder': 'results/miniMNIST'})

##
# Or alternatively use a toml_config file. Notice that you only need to specify the values that you wish to change from the default file.
# You can load the toml file yourself and pass it like this:
with open(os.path.dirname(__file__) + '/classification_train.toml', 'r') as f:
    toml_config = toml.load(f)
run_train(**toml_config)

# Or you can call the script as a module and pass the toml file as a command line arg.
#       python -m src.utils.decoder.train --toml_config_filename src/utils/decoder/examples/classification_train.toml
##



