import argparse
import csv
import math
from pathlib import Path

import numpy as np
import sty
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import os

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    apply_antialiasing,
    add_general_args,
    delete_and_recreate_path,
)
import uuid

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def compute_x(y, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x = (y - b) / m
    return x


class DrawPonzo(DrawStimuli):
    def __init__(self, num_rail_lines=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rail_lines = num_rail_lines

    def get_random_start_end_ponts(self):
        start_point = (
            random.randint(0, self.canvas_size[0]),
            random.randint(0, self.canvas_size[1]),
        )
        end_point = (
            random.randint(0, self.canvas_size[0]),
            random.randint(0, self.canvas_size[1]),
        )
        return start_point, end_point

    def generate_illusory_images(self, same_length=False):
        img = self.create_canvas()
        d = ImageDraw.Draw(img)
        margin = self.canvas_size[0] * 0.11

        ## draw the diagonal
        vertical_distance_from_center = random.randint(
            int(self.canvas_size[0] * 0.11), int(self.canvas_size[0] * 0.33)
        )
        x = self.canvas_size[0] // 2 - vertical_distance_from_center
        length = self.canvas_size[0] - margin * 2
        slope = random.random() * 2 + 3.2  # random.random()
        x2 = x + length // 2 * math.cos(math.atan(-slope))
        y2 = self.canvas_size[1] // 2 + length // 2 * math.sin(math.atan(-slope))
        x1 = x - length // 2 * math.cos(math.atan(-slope))
        y1 = self.canvas_size[1] // 2 - length // 2 * math.sin(math.atan(-slope))
        d1_line_start = (x1, y1)
        d1_line_end = (x2, y2)
        d.line([d1_line_start, d1_line_end], fill="white", width=2)

        x = self.canvas_size[0] // 2 + vertical_distance_from_center
        x2 = x + length // 2 * math.cos(math.atan(slope))
        y2 = self.canvas_size[1] // 2 + length // 2 * math.sin(math.atan(slope))
        x1 = x - length // 2 * math.cos(math.atan(slope))
        y1 = self.canvas_size[1] // 2 - length // 2 * math.sin(math.atan(slope))
        d2_line_start = (x1, y1)
        d2_line_end = (x2, y2)

        d.line([d2_line_start, d2_line_end], fill="white", width=2)

        if self.num_rail_lines > 0:
            vertical_ranges = [
                int(
                    i * (self.canvas_size[1] - margin * 2) // (self.num_rail_lines)
                    + margin
                )
                for i in range(self.num_rail_lines + 1)
            ]
            vertical_line_position = [
                random.randint(vertical_ranges[i], vertical_ranges[i + 1] - 1)
                for i in range(len(vertical_ranges) - 1)
            ]
            for v in vertical_line_position:
                h_start = compute_x(v, d1_line_start, d1_line_end)
                h_end = compute_x(v, d2_line_start, d2_line_end)
                additional = (h_end - h_start) * 0.1
                d.line(
                    ((h_start - additional, v), (h_end + additional, v)),
                    fill="white",
                    width=2,
                )
        ## draw red and line lines
        v_position_up = random.randint(int(margin), self.canvas_size[1] // 2)
        v_position_down = random.randint(
            self.canvas_size[1] // 2, int(self.canvas_size[1] - margin)
        )
        pos = np.array([v_position_down, v_position_up])
        red_length = random.randint(self.canvas_size[0] // 10, self.canvas_size[0] // 2)
        blue_length = (
            red_length
            if same_length
            else random.randint(self.canvas_size[0] // 10, self.canvas_size[0] // 2)
        )
        np.random.shuffle(pos)

        upper_line_color = "red" if pos[0] < pos[1] else "blue"
        d.line(
            (
                (self.canvas_size[0] // 2 - red_length // 2, pos[0]),
                (self.canvas_size[0] // 2 + red_length // 2, pos[0]),
            ),
            fill="red",
            width=2,
        )

        d.line(
            (
                (self.canvas_size[0] // 2 - blue_length // 2, pos[1]),
                (self.canvas_size[0] // 2 + blue_length // 2, pos[1]),
            ),
            fill="blue",
            width=2,
        )

        max_length = self.canvas_size[0] // 2 - self.canvas_size[0] // 10
        label = red_length - blue_length
        norm_label = label / max_length

        return img, label, norm_label, upper_line_color

    def generate_rnd_lines_images(
        self, colored_line_always_horizontal=False, antialias=True
    ):
        img = Image.new("RGB", self.canvas_size, color="black")

        d = ImageDraw.Draw(img)
        for i in range(
            self.num_rail_lines + 2
        ):  # add the diagonal lines which are always present
            start_point, end_point = self.get_random_start_end_ponts()
            d.line([start_point, end_point], fill="white", width=2)

        if colored_line_always_horizontal:
            margin = self.canvas_size[0] * 0.11
            v_position_red = random.randint(
                int(margin), int(self.canvas_size[1] - margin)
            )
            v_position_blue = random.randint(
                int(margin), int(self.canvas_size[1] - margin)
            )
            red_length = random.randint(
                self.canvas_size[0] // 10, self.canvas_size[0] // 2
            )
            blue_length = random.randint(
                self.canvas_size[0] // 10, self.canvas_size[0] // 2
            )
            red_sp = (self.canvas_size[0] // 2 - red_length // 2, v_position_red)
            red_ep = (self.canvas_size[0] // 2 + red_length // 2, v_position_red)
            blue_sp = (self.canvas_size[0] // 2 - blue_length // 2, v_position_blue)
            blue_ep = (self.canvas_size[0] // 2 + blue_length // 2, v_position_blue)

        else:
            red_sp, red_ep = self.get_random_start_end_ponts()
            blue_st, blue_ep = self.get_random_start_end_ponts()

        d.line([red_sp, red_ep], fill="red", width=2)
        red_length = np.linalg.norm(np.array(red_ep) - np.array(red_sp))
        d.line([blue_sp, blue_ep], fill="blue", width=2)
        blue_length = np.linalg.norm(np.array(blue_ep) - np.array(blue_sp))

        max_length = self.canvas_size[0] // 2 - self.canvas_size[0] // 10
        label = red_length - blue_length
        norm_label = label / max_length
        if antialias:
            img = img.resize(tuple(np.array(self.canvas_size) * 2)).resize(
                self.canvas_size, resample=Image.Resampling.LANCZOS
            )
        return apply_antialiasing(img) if self.antialiasing else img, label, norm_label


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "num_samples_scrambled": 5000,
        "num_samples_illusory": 500,
        "num_rail_lines": 5,
        "train_with_rnd_col_lines": False,
        "output_folder": f"data/{category_folder}/{name_dataset}",
    }
)


def generate_all(
    num_samples_scrambled=DEFAULTS["num_samples_scrambled"],
    num_samples_illusory=DEFAULTS["num_samples_illusory"],
    num_rail_lines=DEFAULTS["num_rail_lines"],
    train_with_rnd_col_lines=DEFAULTS["train_with_rnd_col_lines"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    output_folder = Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    [
        (output_folder / i).mkdir(exist_ok=True, parents=True)
        for i in ["scrambled_lines", "ponzo_same_length", "ponzo_diff_length"]
    ]

    ds = DrawPonzo(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        num_rail_lines=num_rail_lines,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "Background",
                "Label",
                "NormalizedLabel",
                "UpperLinEColor",
                "NumRailLines",
                "IterNum",
            ]
        )
        for i in tqdm(range(num_samples_scrambled)):
            img, label, norm_label = ds.generate_rnd_lines_images(
                colored_line_always_horizontal=not train_with_rnd_col_lines,
                antialias=antialiasing,
            )
            path = Path("scrambled_lines") / f"{norm_label:.3f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow(
                [
                    path,
                    "scrambled_lines",
                    ds.background,
                    label,
                    norm_label,
                    "",
                    num_rail_lines,
                    i,
                ]
            )
        for i in tqdm(range(num_samples_illusory)):
            for c in ["ponzo_same_length", "ponzo_diff_length"]:
                img, label, norm_label, upper_line_color = ds.generate_illusory_images(
                    same_length=True
                )
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{norm_label:.3f}_{upper_line_color}_{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        c,
                        ds.background,
                        label,
                        norm_label,
                        upper_line_color,
                        i,
                    ]
                )
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_scrambled",
        "-nscr",
        default=DEFAULTS["num_samples_scrambled"],
        type=int,
    )
    parser.add_argument(
        "--num_samples_illusory",
        "-nill",
        default=DEFAULTS["num_samples_illusory"],
        type=int,
    )

    parser.add_argument(
        "--num_rail_lines",
        "-nrl",
        default=DEFAULTS["num_rail_lines"],
        help="This refers to the number of horizontal lines (excluding the red and blue lines) in the proper illusion shape_based_image_generation. During training, we generate dataset matching the total number of lines, so that this parameter will affect both test and train shape_based_image_generation. Notice that, since in the minimal illusion, two oblique lines are always present, similarly in the train shape_based_image_generation there are always two lines, to which we add a number of lines specified by this parameter",
        type=int,
    )

    parser.add_argument(
        "--train_with_rnd_col_lines",
        "-trc",
        help="Specify whether the red and blue lines in  training dataset should be randomly placed, or should be horizontal like in the testing (illusion) condition",
        action="store_true",
        default=False,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
