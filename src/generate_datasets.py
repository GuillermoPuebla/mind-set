import argparse
from tqdm import tqdm
import toml
import importlib
import os

DEFAULT = {"toml_file": "generate_subset_datasets.toml"}


def generate_toml(toml_file):
    print(toml_file)
    with open(toml_file, "r") as f:
        dataset_params = toml.load(f)

    for dataset, params in tqdm(dataset_params.items()):
        generate_all = getattr(
            importlib.import_module(f"{dataset[:-3].replace(os.sep, '.')}"),
            "generate_all",
        )
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
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
    generate_toml(**args.__dict__)
