import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib
from PIL.ImageOps import grayscale, invert
import PIL.Image as Image
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
from src.utils.compute_distance.misc import get_new_affine_values, my_affine
import re


def get_highest_number(folder_path):
    # List all filenames in the folder
    filenames = os.listdir(folder_path)

    # Initialize the highest number found so far
    highest_number = -1

    for filename in filenames:
        # Extract all sequences of digits from the filename
        numbers = re.findall(r"\d+", filename)

        # If we found any numbers, update the highest number if necessary
        for number_str in numbers:
            number = int(number_str)
            if number > highest_number:
                highest_number = number

    return highest_number


def transform_save(
    original_img, folder, N, affine_str="r[-180, 180]t[-0.2, 0.2]s[0.75, 1.2]"
):
    for i in range(N):
        img = original_img.copy()
        af = get_new_affine_values(affine_str)
        img = my_affine(
            img,
            translate=af["tr"],
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        n = get_highest_number(folder)

        img.save(folder / f"{n+1}.png")


# Takes a path to a figure, load the figure, invert the colors of the figure
def load_and_invert(path):
    img = invert(Image.open(path).convert("L"))
    img = img.resize((224, 224))
    img = img.point(lambda x: 255 if x >= 10 else 0)

    return img


def generate(Ntrain, Ntest):
    left_ds = pathlib.Path("assets/leuven_embedded_figures_test")
    figs_to_take = range(0, 16 * 4, 4)
    i = figs_to_take[0]
    all_shapes_path = [
        left_ds / "shapes" / (str(i).zfill(3) + ".png") for i in figs_to_take
    ]
    all_context_path = [
        left_ds / "context" / (str(i).zfill(3) + "a.png") for i in range(0, 64)
    ]

    output_folder = pathlib.Path("data/gestalt/leuven_embedded_figures_test/")
    output_folder_shape = output_folder / "train"
    [
        (output_folder_shape / str(i)).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(all_shapes_path)
    ]

    ## Train figures
    for idx, s in enumerate(all_shapes_path):
        img = load_and_invert(s)
        transform_save(img, output_folder_shape / str(idx), Ntrain)

    ## Test figures
    # Here we only take the figures containing the "target" shape, not the figures with the distractor. The test consits of checking whether a network can correctly classify these stimuli with a high accuracy.
    # each context path goes in group of 4: from 0 to 4 refer to the 1st shape, from 5 to 8 to the second... So let's group the folders together

    output_folder_context = output_folder / "test"
    [
        shutil.rmtree(output_folder_context / str(i // 4), ignore_errors=True)
        for i, s in enumerate(all_context_path)
    ]
    [
        (output_folder_context / str(i // 4)).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(all_context_path)
    ]
    for idx, s in enumerate(all_context_path):
        img = load_and_invert(s)
        transform_save(img, output_folder_context / str(idx // 4), Ntest // 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Ntrain_per_class",
        default=1000,
    )
    parser.add_argument(
        "--Ntest_per_class",
        default=100,
    )
    args = parser.parse_known_args()[0]
    generate(**args.__dict__)
