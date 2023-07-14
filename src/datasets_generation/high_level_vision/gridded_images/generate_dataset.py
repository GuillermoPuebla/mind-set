import argparse
import csv
import os
import pathlib
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import glob

import sty
from PIL.ImageOps import grayscale, invert

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.compute_distance.misc import (
    paste_at_center,
)
from src.utils.drawing_utils import (
    DrawStimuli,
    resize_image_keep_aspect_ratio,
    paste_linedrawing_onto_canvas,
)
from src.utils.misc import (
    add_general_args,
    add_training_args,
    delete_and_recreate_path,
    apply_antialiasing,
    DEFAULTS,
)


class DrawGriddedImages(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def apply_grid_mask(
        self,
        image_path,
        grid_size,
        grid_thickness=1,
        grid_shift=0,
        rotation_degrees=0,
        complement=False,
    ):
        opencv_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)
        img = np.array(
            paste_linedrawing_onto_canvas(
                opencv_img, self.create_canvas(), self.line_args["fill"]
            )
        )

        height, width, _ = img.shape

        mask = np.full((height * 2, width * 2), False)

        for i in range(grid_shift, mask.shape[0], grid_size):
            mask[i : i + grid_thickness, :] = 1

        rotated_mask = np.array(
            Image.fromarray(mask).rotate(rotation_degrees, expand=True, fillcolor=(0))
        )
        rotated_mask = rotated_mask[
            rotated_mask.shape[0] // 2
            - height // 2 : rotated_mask.shape[0] // 2
            - height // 2
            + height,
            rotated_mask.shape[1] // 2
            - width // 2 : rotated_mask.shape[1] // 2
            - width // 2
            + width,
        ]

        if complement:
            img[~rotated_mask] = self.background[0]
        else:
            img[rotated_mask] = self.background[0]

        img = Image.fromarray(img.astype(np.uint8))
        img = paste_at_center(self.create_canvas(), img)

        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return apply_antialiasing(img) if self.antialiasing else img


def generate_all(
    linedrawing_input_folder,
    object_longest_side,
    grid_degree,
    grid_size,
    grid_thickness,
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    linedrawing_input_folder = (
        pathlib.Path("assets") / "baker_2018" / "outline_images_fix"
        if linedrawing_input_folder is None
        else linedrawing_input_folder
    )

    output_folder = (
        pathlib.Path("data") / "high_level_vision" / "gridded_images"
        if output_folder is None
        else pathlib.Path(output_folder)
    )

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return output_folder

    delete_and_recreate_path(output_folder)

    all_categories = [i.stem for i in linedrawing_input_folder.glob("*")]

    [
        [
            (output_folder / f"{ff}" / cat).mkdir(parents=True, exist_ok=True)
            for ff in ["del", "del_complement"]
            for cat in all_categories
        ]
    ]

    ds = DrawGriddedImages(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        width=1,
    )
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "ClassName",
                "IsComplement",
                "Background",
                "GridShift",
                "GridThickness",
                "GridDegree",
            ]
        )
        grid_shift = 0
        for complement in [True, False]:
            for img_path in linedrawing_input_folder.glob("*"):
                class_name = img_path.stem

                img = ds.apply_grid_mask(
                    img_path,
                    grid_size,
                    grid_shift=grid_shift,
                    grid_thickness=grid_thickness,
                    rotation_degrees=grid_degree,
                    complement=complement,
                )
                path = (
                    pathlib.Path(
                        ("del" + ("_complement" if complement else ""))
                        + "/"
                        + class_name
                    )
                    / f"0.png"
                )

                img.save(str(output_folder / path))
                writer.writerow(
                    [
                        path,
                        class_name,
                        complement,
                        background,
                        grid_shift,
                        grid_thickness,
                        grid_degree,
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=100,
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--folder_linedrawings",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
    )
    parser.add_argument(
        "--grid_degree",
        "-gd",
        type=int,
        default=45,
        help="The rotation of the grid, in angles.",
    )
    parser.add_argument(
        "--grid_size",
        "-gs",
        help="The size of each cell of the grid (in pixels)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--grid_thickness",
        "-gt",
        default=4,
        type=int,
        help="The thickness of the grid (in pixels)",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
