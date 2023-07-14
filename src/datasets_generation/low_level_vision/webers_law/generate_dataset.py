import argparse
from cgi import test
import csv
import math
import os
import errno
import pathlib
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import itertools
import random
from torchvision.transforms import transforms, InterpolationMode
import sty
from torch import rand

from PIL import Image, ImageDraw
import math
from src.utils.compute_distance.misc import get_new_affine_values, my_affine
from src.utils.drawing_utils import DrawStimuli

from src.utils.misc import DEFAULTS, add_general_args, delete_and_recreate_path


def draw_line(length, width_range, lum_range, len_var, len_unit, size_imx, size_imy):
    im = Image.new("RGB", (size_imx, size_imy), color="black")

    ### Randomly draw width and luminance from range
    width = np.random.randint(width_range[0], width_range[1])
    lum = np.random.randint(lum_range[0], lum_range[1])
    delta_len = np.random.randint(len_var[0], len_var[1])
    len_in_pix = length * len_unit
    new_len_in_pix = len_in_pix + delta_len

    ### Find coordinates of line in the middle of image
    xc = size_imx / 2
    yc = size_imy / 2
    x0, y0 = xc - (new_len_in_pix / 2), yc
    x1, y1 = xc + (new_len_in_pix / 2), yc
    bbox = [(x0, y0), (x1, y1)]

    drawing = ImageDraw.Draw(im)
    drawing.line(bbox, width=width, fill=(lum, lum, lum))

    return im


class DrawWeberLength(DrawStimuli):
    def gen_stim(
        self,
        length,
        width,
        lum,
        # len_var,
        # len_unit,
    ):
        img = self.create_canvas()
        x0, y0 = self.canvas_size[0] / 2 - (length / 2), self.canvas_size[1] / 2
        x1, y1 = self.canvas_size[0] / 2 + (length / 2), self.canvas_size[1] / 2
        bbox = [(x0, y0), (x1, y1)]

        drawing = ImageDraw.Draw(img)
        drawing.line(bbox, width=width, fill=(lum, lum, lum))
        return img


DEFAULTS["num_samples_per_length"] = 100
DEFAULTS["max_line_length"] = 50
DEFAULTS["min_line_length"] = 5
DEFAULTS["interval_line_length"] = 1


def generate_all(
    num_samples_per_length=DEFAULTS["num_samples_per_length"],
    max_line_length=DEFAULTS["max_line_length"],
    min_line_length=DEFAULTS["min_line_length"],
    interval_line_length=DEFAULTS["interval_line_length"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
) -> str:
    lengths_conditions = range(min_line_length, max_line_length, interval_line_length)
    output_folder = (
        pathlib.Path("data") / "low_level_vision" / "Webers_Law_length"
        if output_folder is None
        else pathlib.Path(output_folder)
    )

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    [
        (output_folder / str(i)).mkdir(exist_ok=True, parents=True)
        for i in lengths_conditions
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Background", "Length", "Width", "Luminance", "IterNum"]
        )
        ds = DrawWeberLength(
            background=background_color,
            canvas_size=canvas_size,
            antialiasing=antialiasing,
        )

        for l in lengths_conditions:
            for n in range(num_samples_per_length):
                # with a canvas_size_y of 224, this is from 1 to 5 pixels width.
                width = int(np.random.uniform(0.0044, 0.02232) * ds.canvas_size[1])
                lum = int(np.random.uniform(100, 256))

                img_path = pathlib.Path(str(l)) / f"{n}.png"
                img = ds.gen_stim(l, width, lum)
                img.save(output_folder / img_path)

                ## TODO: Notice that we do not apply object transformations. In MindSet, we assume that image transformations are all done somewhere else, e.g. during training, by pycharm (there are some exception to this rule, in those cases where pycharm would screw up the image background). However, here Gaurav found particularly important to separate train/testing by translation location. This is tricky. If I do perform transformation here (and save the location in the annotation file, which then the user can use to separate different transformation for train/test), why not on ALL other datasets? We need to think about this.
                writer.writerow([img_path, ds.background, l, width, lum, n])

    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--num_samples_per_length",
        "-nsl",
        default=DEFAULTS["num_samples_per_length"],
        help="The number of samples to generate for each length condition",
    )
    parser.add_argument(
        "--max_line_length",
        "-maxll",
        default=DEFAULTS["max_line_length"],
        help="The maximum line length (in pixels) to use",
    )

    parser.add_argument(
        "--min_line_length",
        "-minll",
        default=DEFAULTS["min_line_length"],
        help="The minimum line length (in pixels) to use",
    )

    parser.add_argument(
        "--interval_line_length",
        "-ill",
        default=DEFAULTS["interval_line_length"],
        help="The Interval line length to use",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
