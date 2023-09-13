import csv
import os
import pathlib

import numpy as np
import argparse

import sty
from tqdm import tqdm

from src.generate_datasets.low_level_vision.ebbinghaus_illusion.utils import (
    DrawEbbinghaus,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS["num_samples"] = 5000
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    num_samples=DEFAULTS["num_samples"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    [
        (output_folder / i).mkdir(parents=True, exist_ok=True)
        for i in ["scrambled_circles", "small_flankers", "big_flankers"]
    ]

    ds = DrawEbbinghaus(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Category",
                "NormSizeCenterCircle",
                "NormSizeOtherFlankers",
                "NumFlankers",
                "Background",
                "Shift",
            ]
        )
        for i in tqdm(range(num_samples)):
            r_c = np.random.uniform(0.05, 0.2)
            img = ds.create_random_ebbinghaus(
                r_c=r_c,
                n=5,
                flankers_size_range=(0.05, 0.18),
                colour_center_circle=(255, 0, 0),
            )
            path = pathlib.Path("scrambled_circles") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow([path, "scrambled_circles", r_c, "", 5, ds.background, ""])

            number_flankers = 5
            r_c = np.random.uniform(0.08, 0.1)
            r2 = np.random.uniform(0.12, 0.15)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(
                r_c=r_c,
                d=0.02 + (r_c + r2),
                r2=r2,
                n=number_flankers,
                shift=shift,
                colour_center_circle=(255, 0, 0),
            )
            path = pathlib.Path("big_flankers") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow(
                [path, "big_flankers", r_c, r2, number_flankers, ds.background, shift]
            )

            number_flankers = 8
            r_c = np.random.uniform(0.08, 0.1)
            r2 = np.random.uniform(0.02, 0.08)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(
                r_c=r_c,
                d=0.02 + (r_c + r2),
                r2=r2,
                n=number_flankers,
                shift=shift,
                colour_center_circle=(255, 0, 0),
            )
            path = pathlib.Path("small_flankers") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow(
                [path, "small_flankers", r_c, r2, number_flankers, ds.background, shift]
            )
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
