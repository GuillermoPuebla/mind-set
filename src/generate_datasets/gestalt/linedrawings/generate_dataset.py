import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import sty
import PIL.Image as Image
from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os

DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm


class DrawDottedImage(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, b = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # dotted_img = np.ones_like(img) * 255

        canvas = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 100,
        "linedrawing_input_folder": "assets/baker_2018/outline_images_fix/",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    object_longest_side=DEFAULTS["object_longest_side"],
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    linedrawing_input_folder = Path(linedrawing_input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    all_categories = [i.stem for i in linedrawing_input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    ds = DrawDottedImage(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "Background", "IterNum"])
        for n, img_path in enumerate(tqdm(linedrawing_input_folder.glob("*"))):
            class_name = img_path.stem
            img = ds.get_linedrawings(
                img_path,
            )
            path = Path(class_name) / f"{n}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
