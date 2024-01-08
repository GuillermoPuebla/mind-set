from pathlib import Path
import argparse
from tqdm import tqdm
import toml
import inspect

import importlib
import os
import sty

DEFAULT = {"toml_file": "generate_all_datasets.toml"}


def generate_datasets_from_toml(toml_file):
    print("Using TOML file: " + sty.fg.blue + f"{toml_file}" + sty.fg.rs)
    with open(toml_file, "r") as f:
        dataset_params = toml.load(f)

    for dataset, params in tqdm(dataset_params.items()):
        print(sty.fg.red + f"Generating {dataset}" + sty.rs.fg)
        file = params.pop("file")
        module_path = ".".join(Path(file).parent.parts) + "." + "generate_dataset"
        generate_all = getattr(importlib.import_module(module_path), "generate_all")
        # params = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
        generate_all(**params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toml_file",
        "-tomlf",
        default=DEFAULT["toml_file"],
        help="The file containing the datasets to generate with the parameters. Only the specified datasets are gonna be generated. For each dataset, parameters that are not specified will be set to their defaults",
    )

    args = parser.parse_known_args()[0]
    generate_datasets_from_toml(**args.__dict__)
