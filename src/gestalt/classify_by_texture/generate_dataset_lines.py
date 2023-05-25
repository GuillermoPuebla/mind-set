import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import math
import src.utils.create_training_from_linedrawings as create_training
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image, ImageDraw, ImageFont

from src.utils.compute_distance.misc import get_new_affine_values, my_affine
from src.utils.drawing_utils import DrawShape, get_mask_from_linedrawing
from src.utils.misc import add_general_args, add_training_args


class DrawPatternedCanvas(DrawShape):
    def add_line_pattern(self, canvas, line_length=10, slope_rad=0, density=1):
        width, height = canvas.size

        draw = ImageDraw.Draw(canvas)

        line_segment = [
            (0, 0),
            (line_length * math.cos(slope_rad), line_length * math.sin(slope_rad)),
        ]
        horizontal_spacing = int(np.round(line_length / density))
        vertical_spacing = int(np.round(line_length / density))
        noise = 1
        for y in range(
            0,
            height,
            np.round(
                line_length * np.abs(math.sin(slope_rad)) + vertical_spacing
            ).astype(int),
        ):
            for x in range(
                0,
                width,
                np.round(line_length * math.cos(slope_rad) + horizontal_spacing).astype(
                    int
                ),
            ):
                x_offset, y_offset = random.randint(-noise, noise), random.randint(
                    -noise, noise
                )
                shifted_line_segment = [
                    (point[0] + x + x_offset, point[1] + y + y_offset)
                    for point in line_segment
                ]
                draw.line(shifted_line_segment, **self.line_args)
        return canvas

    def __init__(self, obj_size_ratio, transform_code, density, *args, **kwargs):
        self.obj_size_ratio = obj_size_ratio
        self.density = density
        self.transform__code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_size = tuple(
            (np.array(self.obj_size_ratio) * self.canvas_size).astype(int)
        )

    @DrawShape.resize_up_down
    def draw_pattern(self, img_path, slope_rad, line_length):
        expand_factor = 1.5
        mask = get_mask_from_linedrawing(img_path, self.obj_size)

        canvas = self.add_line_pattern(
            self.create_canvas(
                size=tuple((np.array(self.canvas_size) * expand_factor).astype(int))
            ),
            line_length=line_length,
            slope_rad=slope_rad,
            density=self.density,
        )
        perpendicular_radian = perpendicular_radian = (
            (slope_rad + math.pi / 2)
            if abs(slope_rad + math.pi / 2) <= math.pi / 2
            else (slope_rad - math.pi / 2)
        )
        canvas_foreg_text = self.add_line_pattern(
            self.create_canvas().resize(
                tuple((np.array(self.obj_size) * self.canvas_size).astype(int))
            ),
            line_length,
            perpendicular_radian,
            self.density,
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        canvas.show()

        af = get_new_affine_values(self.transform__code)
        img = my_affine(
            canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        img = transforms.CenterCrop(self.canvas_size[0])(img)
        return img


def generate_all(
    linedrawing_input_folder,
    output_folder,
    canvas_size,
    background,
    num_training_samples,
    num_testing_samples,
    antialiasing,
    object_size,
    density,
):
    transf_code = f"t[-0.1,0.1]"
    linedrawing_input_folder = (
        Path("assets/baker_2018/outline_images_fix/")
        if linedrawing_input_folder is None
        else linedrawing_input_folder
    )

    output_folder = (
        Path("data/gestalt/classify_by_texture")
        if output_folder is None
        else output_folder
    )
    if num_training_samples:
        create_training.generate_all(
            str(linedrawing_input_folder),
            str(output_folder / "train"),
            canvas_size,
            background,
            antialiasing,
            object_size,
            num_training_samples,
            # we use the default transform code
        )

    all_categories = [
        os.path.splitext(os.path.basename(i))[0]
        for i in glob.glob(str(linedrawing_input_folder) + "/**")
    ]

    [
        shutil.rmtree(output_folder / "test" / cat)
        for cat in all_categories
        if (output_folder / "test" / cat).exists()
    ]

    [
        (output_folder / "test" / cat).mkdir(exist_ok=True, parents=True)
        for cat in all_categories
    ]
    ds = DrawPatternedCanvas(
        background=background,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_size_ratio=np.array(object_size) / np.array(canvas_size),
        density=density,
        width=1,
        transform_code=transf_code,  # np.max((1, np.round(canvas_size[0] * 0.00446).astype(int))),
    )
    for f in glob.glob(str(linedrawing_input_folder) + "/**"):
        print(f)
        for n in range(num_testing_samples):
            img = ds.draw_pattern(
                f,
                np.deg2rad(random.uniform(*random.choice([(-60, 60)]))),
                random.randint(4, 8),
            )
            # We don't perform scaling or rotation because it creates many artifacts that make the task too difficult even for humans.
            img.save(
                output_folder
                / "test"
                / os.path.splitext(os.path.basename(f))[0]
                / f"{n}.png"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    add_training_args(parser)
    parser.set_defaults(num_training_samples=0)
    parser.set_defaults(antialiasing=False)

    parser.add_argument(
        "--density",
        "-d",
        default=1.8,
        help="The desity of the pattern. The horizontal and vertical spacing are equal to line_length/density",
    )

    parser.add_argument(
        "--object_size",
        "-objsize",
        default="100x100",
        help="A string in the format NxM specifying the size of object (in pixels) embedded into the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")]),
    )
    parser.add_argument(
        "--folder_linedrawings",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default="assets/baker_2018/outline_images_fix/",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


####

from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import Image, ImageDraw, ImageFont


import cv2
import numpy as np


# Call the functions to create and rotate a canvas
# create_canvas(500, 500, "A", 20, 45)


##
