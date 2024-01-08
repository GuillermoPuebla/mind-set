import argparse
import csv

import sty
import toml
import inspect

from tqdm import tqdm
from .utils import DrawEmergentFeaturesdots
import pathlib

from src.utils.misc import add_general_args, delete_and_recreate_path

from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os
import uuid

DEFAULTS = BASE_DEFAULTS.copy()
DEFAULTS["num_samples"] = 1000
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    num_samples=DEFAULTS["num_samples"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    ds = DrawEmergentFeaturesdots(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=10,
    )

    all_types = ["single", "proximity", "orientation", "linearity"]

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [
        [(output_folder / i / j).mkdir(exist_ok=True, parents=True) for i in all_types]
        for j in ["a", "b"]
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "PairA/B", "SampleId"])
        for i in tqdm(range(num_samples)):
            all = ds.get_all_sets()[0]
            for t in tqdm(all_types, leave=False):
                for ip, pair in enumerate(["a", "b"]):
                    path = pathlib.Path(t) / pair / f"{i}.png"
                    all[t][ip].save(output_folder / path)
                    writer.writerow([path, t, ds.background, pair, i])
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=DEFAULTS["num_samples"],
        help="Each `sample` corresponds to an entire set of pair of shape_based_image_generation, for each condition.",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
