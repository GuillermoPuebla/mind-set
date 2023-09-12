"""
This file generates a toml parameters file containing the parameters for each dataset found in the src/generate_datasetss folder (that is, each file matching the path  src/generate_datasets/**/generate_dataset**.py). The toml parameters file is in the format that can be read by the `generate_datasets.py` function. A user is supposed to change the resulting toml file, not the defaults parameters in the individual source files. 
"""

import importlib
import toml
import os
from tqdm import tqdm
import sty


def create_config(datasets):
    config = {}

    for dataset in tqdm(datasets):
        module = importlib.import_module(f"{dataset[:-3].replace(os.sep, '.')}")
        for k, v in module.DEFAULTS.items():
            if v is None:
                print(
                    sty.fg.red
                    + f"Dataset {dataset}\nWarning: the parameter {k} has value None. None are not supported by toml file and the parameter won't be saved. We suggest to never use None as a default parameter"
                    + sty.rs.fg
                )
        config[dataset] = module.DEFAULTS

    # Write the config to a JSON file
    with open("generate_all_datasets.toml", "w") as f:
        toml.dump(config, f)


import glob

if __name__ == "__main__":
    datasets = glob.glob(
        "src/generate_datasets/**/generate_dataset**.py", recursive=True
    )

    create_config(datasets)
