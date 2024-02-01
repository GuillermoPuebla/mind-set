import argparse
import csv
from pathlib import Path
import uuid

import cv2
import numpy as np
import sty
import PIL.Image as Image
import toml
import inspect

from src.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_and_paste,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
    get_affine_rnd_fun,
    my_affine,
)
from PIL import ImageOps
from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os
from torchvision.transforms import InterpolationMode, transforms


DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm


class DrawTransform(DrawStimuli):
    def get_image_transformed(self, image_path, tr, rt, sc, sh):
        img = Image.fromarray(
            resize_image_keep_aspect_ratio(
                np.array(Image.open(image_path)), max(self.canvas_size)
            )
        )

        canvas = self.create_canvas()
        canvas.paste(
            img,
            (
                canvas.size[0] // 2 - img.size[0] // 2,
                canvas.size[1] // 2 - img.size[1] // 2,
            ),
        )

        canvas = my_affine(
            canvas,
            translate=tr,
            angle=rt,
            scale=sc,
            shear=sh,
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        canvas = ImageOps.invert(canvas.convert("L"))

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "input_folder": "assets/baker_2018_linedrawings/cropped/",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "translation_X": (-0.2, 0.2),
        "translation_Y": (-0.2, 0.2),
        "scale": (0.5, 0.9),
        "rotation": (0, 360),
        "num_samples": 5,
        "background_color": (255, 255, 255),
    }
)


def generate_all(
    input_folder=DEFAULTS["input_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    translation_X=DEFAULTS["translation_X"],
    translation_Y=DEFAULTS["translation_Y"],
    scale=DEFAULTS["scale"],
    rotation=DEFAULTS["rotation"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    num_samples=DEFAULTS["num_samples"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))
    ds = DrawTransform(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
    )

    all_categories = [i.stem for i in input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    jpg_files = list(input_folder.rglob("*.jpg"))
    png_files = list(input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "BackgroundColor",
                "Translation_X",
                "Translation_Y",
                "Rotation",
                "Scale",
                "Shear",
                "IterNum",
            ]
        )
        af = get_affine_rnd_fun(
            {
                "translation_X": translation_X,
                "translation_Y": translation_Y,
                "rotation": rotation,
                "scale": scale,
            }
        )

        for _, img_path in enumerate(tqdm(image_files)):
            for i in range(num_samples):
                rnd_v = af()
                trX, trY, rt, sc, sh = (
                    rnd_v["tr"][0],
                    rnd_v["tr"][1],
                    rnd_v["rt"],
                    rnd_v["sc"],
                    rnd_v["sh"],
                )

                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.get_image_transformed(img_path, [trX, trY], rt, sc, sh)
                uui = str(uuid.uuid4().hex[:8])
                path = Path(class_name) / f"{image_name}_{uui}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        class_name,
                        ds.background,
                        rnd_v["tr"][0],
                        rnd_v["tr"][1],
                        rnd_v["rt"],
                        rnd_v["sc"],
                        rnd_v["sh"],
                        i,
                    ]
                )

    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(background_color=DEFAULTS["background_color"])

    parser.add_argument(
        "--num_samples",
        "-ns",
        default=DEFAULTS["num_samples"],
        type=int,
        help="Number of transformation for each object",
    )
    parser.add_argument(
        "--translation_X",
        "-trX",
        default=DEFAULTS["translation_X"],
        type=lambda x: (tuple([float(i) for i in x.split("_")]) if "_" in x else x),
        help="Maximum absolute fraction for horizontal translation, from -1 to 1. From commandline, use MIN_MAX.",
    )
    parser.add_argument(
        "--translation_Y",
        "-trY",
        default=DEFAULTS["translation_Y"],
        type=lambda x: (tuple([float(i) for i in x.split("_")]) if "_" in x else x),
        help="Maximum absolute fraction for vertical translation, from -1 to 1. From commandline, use MIN_MAX.",
    )
    parser.add_argument(
        "--scale",
        "-sc",
        default=DEFAULTS["scale"],
        type=lambda x: (tuple([float(i) for i in x.split("_")]) if "_" in x else x),
        help="Scaling factor range, where 1 is the original scale. From commandline, use MIN_MAX.",
    )
    parser.add_argument(
        "--rotation",
        "-rot",
        default=DEFAULTS["rotation"],
        type=lambda x: (tuple([float(i) for i in x.split("_")]) if "_" in x else x),
        help="Rotation range in degree. From commandline, use MIN_MAX",
    )

    parser.add_argument(
        "--input_folder",
        "-if",
        dest="input_folder",
        help="A folder of input images",
        default=DEFAULTS["input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
