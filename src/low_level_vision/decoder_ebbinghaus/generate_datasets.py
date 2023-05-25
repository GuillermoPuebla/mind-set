import glob
import os
import pathlib
import shutil

import numpy as np
import argparse

from src.low_level_vision.decoder_ebbinghaus.utils import DrawEbbinghaus
from src.utils.misc import add_general_args, add_training_args


def generate_all(
    output_folder,
    canvas_size,
    background,
    num_training_samples,
    num_testing_samples,
    antialiasing,
):
    output_folder = (
        pathlib.Path("data/low_level_vision/decoder_ebbinghaus/")
        if output_folder is None
        else output_folder
    )
    [
        shutil.rmtree(f)
        for f in glob.glob(str(output_folder / "**"))
        if os.path.exists(f)
    ]

    (output_folder / "train").mkdir(parents=True, exist_ok=True)
    [
        (output_folder / "test" / i).mkdir(parents=True, exist_ok=True)
        for i in ["like_train", "small_flankers", "big_flankers"]
    ]

    ds = DrawEbbinghaus(
        background=background, canvas_size=canvas_size, antialiasing=antialiasing
    )

    for i in range(num_training_samples):
        r_c = np.random.uniform(0.05, 0.2)
        img = ds.create_random_ebbinghaus(
            r_c=r_c,
            n=5,
            flankers_size_range=(0.05, 0.18),
            colour_center_circle=(255, 0, 0),
        )
        img.save(output_folder / "train" / f"{r_c}_{i}.png")
        img.save(output_folder / "test" / "like_train" / f"{r_c:.5f}_{i}.png")

    n_small = 8
    n_large = 5
    for i in range(num_testing_samples):
        r_c = np.random.uniform(0.08, 0.1)
        r2 = np.random.uniform(0.12, 0.15)
        shift = np.random.uniform(0, np.pi)
        img = ds.create_ebbinghaus(
            r_c=r_c,
            d=0.02 + (r_c + r2),
            r2=r2,
            n=n_large,
            shift=shift,
            colour_center_circle=(255, 0, 0),
        )
        img.save(output_folder / "test" / "big_flankers" / f"{r_c:.5f}_{i}.png")

        r_c = np.random.uniform(0.08, 0.1)
        r2 = np.random.uniform(0.02, 0.08)
        shift = np.random.uniform(0, np.pi)
        img = ds.create_ebbinghaus(
            r_c=r_c,
            d=0.02 + (r_c + r2),
            r2=r2,
            n=n_small,
            shift=shift,
            colour_center_circle=(255, 0, 0),
        )
        img.save(output_folder / "test" / "small_flankers" / f"{r_c:.5f}_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    add_training_args(parser)

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
