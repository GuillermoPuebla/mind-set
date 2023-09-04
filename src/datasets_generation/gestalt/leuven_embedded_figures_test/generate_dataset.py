import argparse
import csv
import os
import shutil
import pathlib

import sty
from PIL.ImageOps import invert
from PIL import Image
from torchvision.transforms import InterpolationMode
from src.utils.similarity_judgment.misc import get_new_affine_values, my_affine
import re
from tqdm import tqdm
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)
from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()
"""
Using the dataset from Downloaded from https://figshare.com/articles/dataset/Leuven_Embedded_Figures_Test_Target_Shapes/3807885 (shapes) and here https://figshare.com/articles/dataset/Leuven_Embedded_Figures_Test_Context_Shapes/3807894 (embedded shapes). These dataset should already be in the assets/leuven_embedded_figures_test folder 
"""


def get_highest_number(folder_path):
    filenames = os.listdir(folder_path)
    highest_number = -1

    for filename in filenames:
        numbers = re.findall(r"\d+", filename)

        for number_str in numbers:
            number = int(number_str)
            if number > highest_number:
                highest_number = number

    return highest_number


def load_and_invert(path, canvas_size, background, antialiasing):
    img = invert(Image.open(path).convert("RGB"))
    img = img.resize(canvas_size)
    img = img.point(lambda x: 255 if x >= 10 else 0)

    img = img.convert("RGB")

    data = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = data[x, y]
            if (
                r == 0 or g == 0 or b == 0
            ):  # If the pixel is not black (>=0) in the grayscale image
                data[x, y] = background

    return apply_antialiasing(img) if antialiasing else img


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    left_ds = pathlib.Path("assets") / "leuven_embedded_figures_test"

    figs_to_take = range(0, 16 * 4, 4)
    i = figs_to_take[0]
    all_shapes_path = [
        left_ds / "shapes" / (str(i).zfill(3) + ".png") for i in figs_to_take
    ]
    all_context_path = [
        left_ds / "context" / (str(i).zfill(3) + "a.png") for i in range(0, 64)
    ]

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    #
    output_folder_shape = output_folder / "shapes"
    [
        (output_folder_shape / str(i)).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(all_shapes_path)
    ]
    output_folder_context = output_folder / "context"

    [
        (output_folder_context / str(i // 4)).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(all_context_path)
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "Class", "Background"])
        ## Train figures
        for idx, s in tqdm(enumerate(all_shapes_path)):
            img = load_and_invert(s, canvas_size, background_color, antialiasing)
            folder = output_folder_shape / str(idx)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow(
                [f"shapes/{str(idx)}/{n + 1}.png", "shapes", idx, background_color]
            )

            ## Test figures
            # Here we only take the figures containing the "target" shape, not the figures with the distractor (which are also provided in the dataset). That's because the standard test consits of checking whether a network can correctly classify these shape_based_image_generation with a high accuracy.
            # each context path goes in group of 4: from 0 to 4 refer to the 1st shape, from 5 to 8 to the second shape, etc.

        for idx, s in enumerate(tqdm(all_context_path, leave=False)):
            img = load_and_invert(s, canvas_size, background_color, antialiasing)
            folder = output_folder_context / str(idx // 4)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow(
                [
                    f"context/{str(idx // 4)}/{n + 1}.png",
                    "context",
                    idx // 4,
                    background_color,
                ]
            )
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
