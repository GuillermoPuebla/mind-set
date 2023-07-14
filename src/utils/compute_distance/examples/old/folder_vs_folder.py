"""
In this script we show 3 ways to compute distances between samples across folders. All samples from the "{folder}/{base_name}" folder will be compared from samples with the same name across all other folders. See README for more details.
"""

import os
import toml
from src.utils.compute_distance.run import compute_distance


# Firstly, we can use just few training options. The options in "default_distance_config" will be used for all unspecified options.
# Notice that in this way you can programmatically generate config file. E.g. here we run two different computation session, with the two different pretraining options. This will create two different toml_config file. Remember to specify a different result folder as in this example.
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        input_paths=dict(
            reference_name="data/examples/NAPvsMPlines/NS",
            folder="data/examples/NAPvsMPlines",
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/examples/distance_similarity/{p}/NAPvsMPlines/"
        ),
        transformation=dict(repetitions=5),
    )


# Or alternatively use a toml_config file. Notice that you only need to specify the values that you wish to change from the default file.
# You can load the toml file yourself and pass it like this:
with open(os.path.dirname(__file__) + "/image_vs_folder.toml", "r") as f:
    toml_config = toml.load(f)
compute_distance(**toml_config)


# Or you can call the script as a module and pass the toml file as a command line arg.
#       python -m src.utils.compute_distance.run --toml_config_filename src/utils/compute_distance/examples/image_vs_folder.toml
##
