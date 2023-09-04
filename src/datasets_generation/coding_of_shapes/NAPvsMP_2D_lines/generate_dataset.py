import argparse
import csv
import json
import pathlib
import PIL.Image as Image
import sty
from tqdm import tqdm

import os
from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


class DrawLines(DrawStimuli):
    def process_image(self, image_path, line_color):
        self.create_canvas()  # dummy call to update the background color
        img = Image.open(image_path)

        colors = {
            (0, 0, 0): self.background,
            (255, 255, 255): line_color,
        }

        img = img.convert("RGB")
        data = img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if data[x, y] in colors:
                    data[x, y] = colors[data[x, y]]

        img = img.resize(self.canvas_size)
        return apply_antialiasing(img) if self.antialiasing else img


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
) -> str:
    kubilius_dataset = pathlib.Path("assets") / "kubilius_2017" / "png"
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
        line_color = (255, 255, 255)
        ds = DrawLines(
            background=background_color,
            canvas_size=canvas_size,
            antialiasing=antialiasing,
        )
        for t in tqdm(all_types):
            for i in tqdm((kubilius_dataset / t).glob("*"), leave=False):
                name_sample = i.stem
                img_path = pathlib.Path(t) / f"{name_sample}.png"
                img = ds.process_image(
                    kubilius_dataset / img_path,
                    line_color,
                )
                img.save(output_folder / img_path)
                writer.writerow([img_path, t, ds.background, name_sample])
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
