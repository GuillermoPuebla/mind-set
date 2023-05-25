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
    def get_canvas_char_pattered(
        self,
        size,
        tile_char,
        font_size,
        spacing=0,
        rotation_angle=45,
        font_path="arial.ttf",
    ):
        font = ImageFont.truetype(font_path, font_size)

        if not self.antialiasing:
            img = self.create_canvas(
                size=tuple(
                    [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
                )
            )

            bitmap_mask = self.create_canvas(
                background=(0, 0, 0)
                if self.line_args["fill"] == (255, 255, 255, 255)
                else (0, 0, 0),
                size=tuple(
                    [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
                ),
            )

            bitmap_mask = bitmap_mask.convert("1")

            stroke = self.create_canvas(
                background=self.line_args["fill"],
                size=tuple(
                    [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
                ),
            )

        else:
            bitmap_mask = self.create_canvas(
                size=tuple(
                    [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
                )
            )

        draw = ImageDraw.Draw(bitmap_mask)
        width, height = bitmap_mask.size
        num_x = width // (font_size + spacing)
        num_y = height // (font_size + spacing)

        for i in range(num_x):
            for j in range(num_y):
                draw.text(
                    (i * font_size + i * spacing, j * font_size + j * spacing),
                    tile_char,
                    fill=self.line_args["fill"][0]
                    if bitmap_mask.mode == "1"
                    else self.line_args["fill"],
                    font=font,
                )
        if not self.antialiasing:
            img.paste(stroke, mask=bitmap_mask)
        else:
            img = bitmap_mask
        # img.show()
        if rotation_angle > 0:
            img = img.rotate(rotation_angle, resample=Image.Resampling.NEAREST)

        img = transforms.CenterCrop(size[0])(img)
        return img

    def __init__(self, obj_size_ratio, transform_code, *args, **kwargs):
        self.obj_size_ratio = obj_size_ratio
        self.transform_code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_size = tuple(
            (np.array(self.obj_size_ratio) * self.canvas_size).astype(int)
        )

    # @DrawShape.resize_up_down
    # We don't use the resize_up_down technique because antialiasing is already obtained in a different way (to account for the "char")
    def draw_pattern(
        self,
        img_path,
        background_char,
        background_font_size,
        rotation_angle_rad,
        background_spacing,
        foreground_char,
        foreground_font_size,
        foreground_spacing,
    ):
        expand_factor = 1.5
        mask = get_mask_from_linedrawing(img_path, self.obj_size)

        canvas = self.get_canvas_char_pattered(
            size=tuple(np.array(self.canvas_size) * expand_factor),
            tile_char=background_char,
            font_size=background_font_size,
            rotation_angle=np.rad2deg(rotation_angle_rad),
            spacing=background_spacing,
        )

        perpendicular_radian = perpendicular_radian = (
            (rotation_angle_rad + math.pi / 2)
            if abs(rotation_angle_rad + math.pi / 2) <= math.pi / 2
            else (rotation_angle_rad - math.pi / 2)
        )
        # perpendicular_radian = 0
        canvas_foreg_text = self.get_canvas_char_pattered(
            size=self.obj_size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=foreground_spacing,
        )
        # canvas_foreg_text.show()

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        # canvas.show()

        af = get_new_affine_values(self.transform_code)
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


def get_spacing(char):
    if char in [".", "*", '"']:
        return -10
    else:
        return 0


def generate_all(
    linedrawing_input_folder,
    output_folder,
    canvas_size,
    background,
    num_training_samples,
    num_testing_samples,
    antialiasing,
    object_size,
    background_char,
    foreground_char,
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
        width=1,
        transform_code=transf_code,
    )

    for f in glob.glob(str(linedrawing_input_folder) + "/**"):
        print(f)
        for n in range(num_testing_samples):
            font_size = random.randint(15, 20)
            img = ds.draw_pattern(
                img_path=f,
                background_char=background_char,
                foreground_char=foreground_char,
                background_font_size=font_size,
                foreground_font_size=font_size,
                rotation_angle_rad=np.deg2rad(
                    random.uniform(*random.choice([(-60, 60)]))
                ),
                foreground_spacing=get_spacing(foreground_char),
                background_spacing=get_spacing(background_char),
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
    # parser.set_defaults(antialiasing=False)

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

    parser.add_argument(
        "--background_char",
        "-bgch",
        default="\\",
        help="The character to be used as background",
    )

    parser.add_argument(
        "--foreground_char",
        "-fgch",
        default=".",
        help="The character to be used as foreground",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


###
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch \ -fgch .
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch . -fgch ""
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch "" -fgch .
