import argparse
import csv
import os
import random

import cv2
import numpy as np
from pathlib import Path

import sty
from PIL import Image, ImageDraw
import math
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.utils.similarity_judgment.misc import get_affine_rnd_fun_from_code, my_affine
from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


class DrawPatternedCanvas(DrawStimuli):
    def __init__(self, obj_longest_side, transform_code, *args, **kwargs):
        self.transform_code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_canvas_char_pattered(
        self,
        size,
        tile_char,
        font_size,
        spacing=0,
        rotation_angle=45,
        font_path="assets/arial.ttf",
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

        img = transforms.CenterCrop((size[1], size[0]))(img)
        return img

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
        opencv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        mask = get_mask_from_linedrawing(opencv_img, fill=True)

        canvas = self.get_canvas_char_pattered(
            size=tuple(np.array(self.canvas_size) * expand_factor),
            tile_char=background_char,
            font_size=background_font_size,
            rotation_angle=np.rad2deg(rotation_angle_rad),
            spacing=background_spacing,
        )

        perpendicular_radian = (
            (rotation_angle_rad + math.pi / 2)
            if abs(rotation_angle_rad + math.pi / 2) <= math.pi / 2
            else (rotation_angle_rad - math.pi / 2)
        )
        canvas_foreg_text = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=foreground_spacing,
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )

        af = get_affine_rnd_fun_from_code(self.transform_code)()
        img = my_affine(
            canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return apply_antialiasing(img) if self.antialiasing else img


def get_spacing(char):
    if char in [".", "*", '"']:
        return -10
    else:
        return 0


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS.update(
    {
        "linedrawing_input_folder": "assets/baker_2018/outline_images_fix/",
        "num_samples": 500,
        "object_longest_side": 200,
        "background_char": "\\",
        "foreground_char": ".",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    num_samples=DEFAULTS["num_samples"],
    object_longest_side=DEFAULTS["object_longest_side"],
    background_char=DEFAULTS["background_char"],
    foreground_char=DEFAULTS["foreground_char"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    transf_code = {"translation": [-0.1, 0.1]}
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

    ds = DrawPatternedCanvas(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        width=1,
        transform_code=transf_code,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "Background",
                "BackgroundChar",
                "ForegroundChar",
                "FontSize",
                "RotationAngle",
                "ForegroundSpacing",
                "BackgroundSpacing",
            ]
        )
        for img_path in tqdm(linedrawing_input_folder.glob("*")):
            print(img_path)
            class_name = img_path.stem
            for n in tqdm(range(num_samples), leave=False):
                rotation_angle = random.uniform(*random.choice([(-60, 60)]))
                font_size = random.randint(15, 20)
                foreground_spacing = get_spacing(foreground_char)
                background_spacing = get_spacing(background_char)
                img = ds.draw_pattern(
                    img_path=img_path,
                    background_char=background_char,
                    foreground_char=foreground_char,
                    background_font_size=font_size,
                    foreground_font_size=font_size,
                    rotation_angle_rad=np.deg2rad(rotation_angle),
                    foreground_spacing=foreground_spacing,
                    background_spacing=background_spacing,
                )
                # We don't perform scaling or rotation because it creates many artifacts that make the task too difficult even for humans.
                path = Path(class_name) / f"{n}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        class_name,
                        ds.background,
                        background_char,
                        foreground_char,
                        font_size,
                        rotation_angle,
                        foreground_spacing,
                        background_spacing,
                    ]
                )
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--num_samples",
        "-ns",
        default=DEFAULTS["num_samples"],
        help="The number of augmented samples to generate for each line drawings",
        type=int,
    )

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio),  before pasting the image into a canvas",
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )

    parser.add_argument(
        "--background_char",
        "-bgch",
        default=DEFAULTS["background_char"],
        help="The character to be used as background",
    )

    parser.add_argument(
        "--foreground_char",
        "-fgch",
        default=DEFAULTS["foreground_char"],
        help="The character to be used as foreground",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


###
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch \ -fgch .
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch . -fgch ""
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch "" -fgch .
