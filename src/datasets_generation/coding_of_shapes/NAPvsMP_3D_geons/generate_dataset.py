import argparse
import colorsys
import csv

import numpy as np
import sty
from PIL import ImageColor
from tqdm import tqdm

from src.datasets_generation.gestalt.CSE_CIE_dots.utils import DrawCSE_CIEdots
import pathlib
import PIL.Image as Image
import os

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()
DEFAULTS["stroke_color"] = ""
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


class DrawLines(DrawStimuli):
    def process_image(self, image_path, shape_color_rgb):
        img = Image.open(image_path).convert("RGB")
        new_img = self.create_canvas(size=img.size)

        colors = {(0, 0, 0): self.background}
        if shape_color_rgb:
            shape_color_hls = colorsys.rgb_to_hls(
                *tuple(np.array(shape_color_rgb) // 255)
            )

        data = img.load()
        new_data = new_img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if data[x, y] in colors:
                    # Change the background color
                    new_data[x, y] = colors[data[x, y]]
                elif shape_color_rgb:
                    # Change the shape color while preserving the lightness
                    pixel_rgb = [v / 255 for v in data[x, y]]  # convert to range [0, 1]
                    pixel_hls = colorsys.rgb_to_hls(*pixel_rgb)
                    new_hls = (
                        shape_color_hls[0],
                        pixel_hls[1],
                        shape_color_hls[2],
                    )
                    new_rgb = [int(v * 255) for v in colorsys.hls_to_rgb(*new_hls)]
                    new_data[x, y] = tuple(new_rgb)
                else:
                    new_data[x, y] = data[x, y]

        new_img = new_img.resize(self.canvas_size)
        return apply_antialiasing(new_img) if self.antialiasing else new_img


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    stroke_color=DEFAULTS["stroke_color"],
    regenerate=DEFAULTS["regenerate"],
) -> str:
    shape_folder = pathlib.Path("assets") / "amir_geons" / "NAPvsMP"
    all_types = ["reference", "MP", "NAP"]
    output_folder = pathlib.Path(output_folder)
    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)
    delete_and_recreate_path(output_folder)
    [(output_folder / i).mkdir(exist_ok=True, parents=True) for i in all_types]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "Background", "SampleName"])
        ds = DrawLines(
            background=background_color,
            canvas_size=canvas_size,
            antialiasing=antialiasing,
        )

        for t in tqdm(all_types):
            for i in tqdm((shape_folder / t).glob("*"), leave=False):
                name_sample = i.stem
                img_path = pathlib.Path(t) / f"{name_sample}.png"
                img = ds.process_image(
                    shape_folder / img_path,
                    stroke_color,
                )
                img.save(output_folder / img_path)
                writer.writerow([img_path, t, ds.background, name_sample])
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--stroke_color",
        "-sc",
        default=DEFAULTS["stroke_color"],
        help="Specify the color of the shape. The shading will be preserved. Leave it empty to not change the color of the shape. Specify it as a rgb tuple in the format of 255_255_255",
        type=lambda x: (tuple([int(i) for i in x.split("_")]) if "_" in x else x)
        if isinstance(x, str)
        else x,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
