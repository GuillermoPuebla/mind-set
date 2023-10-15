import argparse
import csv
import os
import pathlib

import cv2
import numpy as np
import PIL.Image as Image

import sty

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.utils.drawing_utils import (
    DrawStimuli,
    resize_image_keep_aspect_ratio,
    paste_linedrawing_onto_canvas,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


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
        img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)
        img = np.array(
            paste_linedrawing_onto_canvas(
                Image.fromarray(img), self.create_canvas(), self.line_args["fill"]
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
        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "linedrawing_input_folder": "assets/baker_2018/outline_images_fix",
        "object_longest_side": 100,
        "grid_degree": 45,
        "grid_size": 8,
        "grid_thickness": 4,
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    object_longest_side=DEFAULTS["object_longest_side"],
    grid_degree=DEFAULTS["grid_degree"],
    grid_size=DEFAULTS["grid_size"],
    grid_thickness=DEFAULTS["grid_thickness"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    linedrawing_input_folder = pathlib.Path(linedrawing_input_folder)

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

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
        for complement in tqdm([True, False]):
            for img_path in linedrawing_input_folder.glob("*"):
                class_name = img_path.stem
                print(class_name)
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
                        ds.background,
                        grid_shift,
                        grid_thickness,
                        grid_degree,
                    ]
                )
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )
    parser.add_argument(
        "--grid_degree",
        "-gd",
        type=int,
        default=DEFAULTS["grid_degree"],
        help="The rotation of the grid, in angles.",
    )
    parser.add_argument(
        "--grid_size",
        "-gs",
        help="The size of each cell of the grid (in pixels)",
        type=int,
        default=DEFAULTS["grid_size"],
    )
    parser.add_argument(
        "--grid_thickness",
        "-gt",
        default=DEFAULTS["grid_thickness"],
        type=int,
        help="The thickness of the grid (in pixels)",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
