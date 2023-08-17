import argparse
import csv
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import sty
from PIL import Image, ImageDraw
import math
import random

from PIL.ImageDraw import Draw
from tqdm import tqdm

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def vector_length(s, theta):
    theta_rad = math.radians(theta)

    use_cosine_ranges = [(0, 45), (135, 180), (180, 225), (315, 360)]

    for range_start, range_end in use_cosine_ranges:
        if range_start <= theta < range_end:
            return 0.5 * s / abs(math.cos(theta_rad))

    return 0.5 * s / abs(math.sin(theta_rad))


class DrawCompletion(DrawStimuli):
    def __init__(self, circle_color, square_color, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circle_color = circle_color
        self.square_color = square_color

    def draw(
        self,
        center_circle,
        center_square,
        radius_circle,
        side_square,
        notched=False,
        top="s",
        notched_proportion=0.3,
    ):
        img = self.create_canvas()
        draw = Draw(img)

        x_s, y_s = center_square

        x_c, y_c = center_circle
        if top == "s":
            draw.ellipse(
                [
                    (x_c - radius_circle, y_c - radius_circle),
                    (x_c + radius_circle, y_c + radius_circle),
                ],
                outline=self.circle_color,
                fill=self.circle_color,
            )
            if notched:
                draw.rectangle(
                    [
                        (
                            x_s
                            - side_square / 2
                            - side_square * notched_proportion * 0.75,
                            y_s
                            - side_square / 2
                            - side_square * notched_proportion * 0.75,
                        ),
                        (
                            x_s
                            + side_square / 2
                            + side_square * notched_proportion * 0.75,
                            y_s
                            + side_square / 2
                            + side_square * notched_proportion * 0.75,
                        ),
                    ],
                    outline=self.background,
                    fill=self.background,
                )
                notched = False

        draw.rectangle(
            [
                (x_s - side_square / 2, y_s - side_square / 2),
                (x_s + side_square / 2, y_s + side_square / 2),
            ],
            outline=self.circle_color if top == "s" else self.square_color,
            fill=self.square_color,
        )

        if top == "c":
            if notched:
                draw.ellipse(
                    [
                        (
                            x_c - radius_circle - radius_circle * notched_proportion,
                            y_c - radius_circle - radius_circle * notched_proportion,
                        ),
                        (
                            x_c + radius_circle + radius_circle * notched_proportion,
                            y_c + radius_circle + radius_circle * notched_proportion,
                        ),
                    ],
                    outline=self.background,
                    fill=self.background,
                )

            draw.ellipse(
                [
                    (x_c - radius_circle, y_c - radius_circle),
                    (x_c + radius_circle, y_c + radius_circle),
                ],
                outline=self.square_color,
                fill=self.circle_color,
            )

        return apply_antialiasing(img) if self.antialiasing else img


DEFAULTS.update(
    {
        "num_samples": 100,
        "circle_color": (255, 255, 255),
        "square_color": (0, 0, 0),
        "output_folder": "data/low_level_vision/contour_completion",
        "background_color": (100, 100, 100),
    }
)


def generate_all(
    num_samples=DEFAULTS["num_samples"],
    circle_color=DEFAULTS["circle_color"],
    square_color=DEFAULTS["square_color"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    ds = DrawCompletion(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        circle_color=circle_color,
        square_color=square_color,
    )

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
        (output_folder / cond).mkdir(exist_ok=True, parents=True)
        for cond in ["occlusion", "no_occlusion", "notched"]
    ]

    top_shapes = ["s", "c"]

    check_square_fully_in_canvas = lambda center_square: (
        center_square[0] - side_square // 2 > 0
        and center_square[0] + side_square // 2 < canvas_size[0]
        and center_square[1] - side_square // 2 > 0
        and center_square[1] + side_square // 2 < canvas_size[1]
    )
    get_center_square = lambda theta, ll: (
        np.cos(theta) * ll + center_circle[0],
        np.sin(theta) * ll + center_circle[1],
    )
    pbar = tqdm(total=num_samples, dynamic_ncols=True)

    completed_samples = 0
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "Background",
                "TopShape",
                "CenterCircle",
                "CenterSquare",
                "RadiusCircle",
                "SideSquare",
                "SampleId",
            ]
        )

        while completed_samples < num_samples:
            radius_circle = random.randint(20, 40)
            side_square = radius_circle * 1.5
            diagonal_square = side_square * np.sqrt(2)

            center_circle = (
                random.randint(0 + radius_circle, canvas_size[1] - radius_circle),
                random.randint(0 + radius_circle, canvas_size[0] - radius_circle),
            )
            theta = np.random.uniform(0, np.pi * 2)

            # Generate unoccluded
            ll = np.random.uniform(
                diagonal_square / 2 + radius_circle,
                np.sqrt(canvas_size[1] ** 2 + canvas_size[0] ** 2),
            )
            center_square = get_center_square(theta, ll)
            if not check_square_fully_in_canvas(center_square):
                continue
            pbar.update(1)
            # note: the top shape is normally the shape "on top" in the occluded version, and the shape with a visible border in the notched and unoccluded version.
            for top_shape in top_shapes:
                img = ds.draw(
                    center_circle,
                    center_square,
                    radius_circle,
                    side_square,
                    notched=False,
                    top=top_shape,
                )
                path = Path("no_occlusion") / f"{top_shape}_{completed_samples}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        "no_occlusion",
                        ds.background,
                        top_shape,
                        center_circle,
                        center_square,
                        radius_circle,
                        side_square,
                        completed_samples,
                    ]
                )

            # Generate occluded and notched
            max_dist_occluded = vector_length(side_square, theta) + radius_circle
            ll = np.random.uniform(radius_circle // 1.2, max_dist_occluded)

            center_square = get_center_square(theta, ll)
            for notched in [True, False]:
                for top_shape in top_shapes:
                    img = ds.draw(
                        center_circle,
                        center_square,
                        radius_circle,
                        side_square,
                        notched=notched,
                        top=top_shape,
                    )
                    path = (
                        Path("notched" if notched else "occlusion")
                        / f"{top_shape}_{completed_samples}.png"
                    )
                    img.save(output_folder / path)
                    writer.writerow(
                        [
                            path,
                            "notched" if notched else "occlusion",
                            ds.background,
                            top_shape,
                            center_circle,
                            center_square,
                            radius_circle,
                            side_square,
                            completed_samples,
                        ]
                    )

            completed_samples += 1
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.set_defaults(background_color=DEFAULTS["background_color"])
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=DEFAULTS["num_samples"],
        help="Each `sample` corresponds to an entire set of pair of shape_based_image_generation, for each condition.",
    )
    parser.add_argument(
        "--circle_color",
        "-ccol",
        default=DEFAULTS["circle_color"],
        help="The color of the circle object.",
        type=lambda x: (tuple([int(i) for i in x.split("_")]) if "_" in x else x),
    )
    parser.add_argument(
        "--square_color",
        "-scol",
        default=DEFAULTS["square_color"],
        help="The color of the square object.",
        type=lambda x: (tuple([int(i) for i in x.split("_")]) if "_" in x else x),
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
