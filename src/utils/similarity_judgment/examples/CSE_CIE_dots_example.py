import pathlib
from pathlib import Path
from src.utils.similarity_judgment.run import compute_distance

dataset_folder = "data/examples/CSE_CIE_dots_small/"

"""
To make this example work, we need the CSE_CIE_dots dataset in the system. If not present, we need to create it. Since this is just an example, we create a small version of it: 
"""

from src.datasets_generation.gestalt.CSE_CIE_dots.generate_dataset import (
    generate_all as CSE_CIE_dots_generate,
)

if not Path(dataset_folder).exists():
    CSE_CIE_dots_generate(output_folder=dataset_folder, num_samples=20)

"""
Now we show 3 ways to compute distances. They all rely on the annotation file being present, and they are _completely_ independent on the actual folder structure. The only requirement is that the annotation file be at the level of the top dataset directory (that is to say: do not move around the annotation.csv file from its location in the dataset!)


--------- APPROACH NUMBER 1: DO EVERYTHING IN PYTHON. --------- 
With this approach, we specify in python, as a dict, all the config options that are different to the  "default_distance_config". This file will be used for all unspecified options.

Notice that in this way you can programmatically generate config file. E.g. here we run two different computation session, with the two different pretraining options. This will create two different toml_config file. Remember to specify a different result folder as in this example.
"""
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        basic_info=dict(
            annotation_file_path=pathlib.Path(dataset_folder) / "annotation.csv",
            match_factors=["Type", "IterNum"],
            factor_variable="PairA/B",
            reference_level="a",
            # (notice that you could limit your analysis to a single level for a factor, by specifying dict(factorName="levelName") here
            filter_factor_level={},  # e.g. dict(Type="single"),
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/examples/distance_similarity/{p}/high_level_vision/CSE_CIE_dots"
        ),
        transformation=dict(affine_transf_code=""),
    )


"""
--------- APPROACH NUMBER 2: WRITE AN ACTUAL TOML FILE, AND PASS IT TO THE FUNCTION --------- 
If you prefer, you can actually write your own config file for this particular task. Notice that you only need to specify the values that you wish to change from the default file.
You can load the toml file yourself and pass it like this: (uncomment this if you wanna use it).
"""
# import os
# import toml
# with open(os.path.dirname(__file__) + "/CSE_CIE_dots_config.toml", "r") as f:
#    toml_config = toml.load(f)
# compute_distance(**toml_config)


"""
--------- APPROACH NUMBER 3: DO EVERYTHING THROUGH COMMAND LINE. --------- 
Yet another way is to  call the script as a module and pass the toml file as a command line arg.
"""
#       python -m src.utils.similarity_judgment.run --toml_config_filename src/utils/compute_distance/examples/CSE_CIE_dots_config.toml
