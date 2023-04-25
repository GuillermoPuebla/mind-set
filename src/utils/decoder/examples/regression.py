"""
In this script we show 3 ways to run a REGRESSION training session.
Before running this example, generate the ebbinghaus dataset in this way:
    python -m src.ebbinghaus.generate_datasets.py --num_training_data=100 --num_testing_data=20 --folder data/examples/ebbinghaus
"""
import os
import toml
from src.utils.decoder.train import run_train

##
# Firstly, we can use just few training options. The options in "default_train_config" will be used for all other options.
data_folder = "data/examples/ebbinghaus"
run_train(
    training=dict(
        train_dataset=f"{data_folder}/random_data_n100/train",
        test_datasets=[
            f"{data_folder}/random_data_n20/test",
            f"{data_folder}/big_flankers_data_n20/test",
            f"{data_folder}/small_flankers_data_n20/test",
        ],
    ),
    saving_folders=dict(result_folder="results/examples/decoder/ebbinghaus"),
)

##
# Or alternatively use a toml_config file. Notice that you only need to specify the values that you wish to change from the default file.
# You can load the toml file yourself and pass it like this:
with open(os.path.dirname(__file__) + "/regression_train.toml", "r") as f:
    toml_config = toml.load(f)
run_train(**toml_config)

# Or you can call the script as a module and pass the toml file as a command line arg.
#       python -m src.utils.decoder.train --toml_config_filename src/utils/decoder/examples/regression_train.toml
##
