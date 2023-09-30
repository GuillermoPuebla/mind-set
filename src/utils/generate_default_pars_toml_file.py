"""
This file generates a toml parameters file containing the parameters for each dataset found in the src/generate_datasetss folder (that is, each file matching the path  src/generate_datasets/**/generate_dataset**.py). The toml parameters file is in the format that can be read by the `generate_datasets.py` function. A user is supposed to change the resulting toml file, not the defaults parameters in the individual source files. 
"""

import importlib
import toml
import os
from tqdm import tqdm
from pathlib import Path
import sty
import glob


def create_config(
    datasets=glob.glob("src/generate_datasets/**/generate_dataset**.py", recursive=True),
    save_to=Path("generate_all_datasets.toml"),
):
    config = {}
    datasets = [Path(dataset) for dataset in datasets]

    for dataset in tqdm(datasets):
        module = ".".join(list(dataset.parts)).strip(".py")
        module = importlib.import_module(module)
        for k, v in module.DEFAULTS.items():
            if v is None:
                print(
                    sty.fg.red
                    + f"Dataset {dataset}\nWarning: the parameter {k} has value None. None are not supported by toml file and the parameter won't be saved. We suggest to never use None as a default parameter"
                    + sty.rs.fg
                )
        config[dataset.as_posix()] = module.DEFAULTS

    # Write the config to a JSON file
    with open(save_to, "w") as f:
        toml.dump(config, f)


if __name__ == "__main__":
    create_config()
