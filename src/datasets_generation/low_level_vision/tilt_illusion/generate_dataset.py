import argparse
import csv
import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import sty
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def generate_grating(canvas_size, frequency, orientation, phase=0):
    width, height = canvas_size
    # Generate x and y coordinates
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    x, y = np.meshgrid(x, y)

    # Rotate the grid by the specified orientation
    x_prime = x * np.cos(orientation) - y * np.sin(orientation)

    # Create the sinusoidal grating
    grating = 0.5 * (1 + np.sin(frequency * x_prime + phase))
    return grating


all_pil_images = []
freq = 10


class DrawTiltIllusion(DrawStimuli):
    def generate_illusion(
        self, theta_test, radius, center_test, freq, theta_context=None
    ):
        if theta_context is not None:
            context = generate_grating(self.canvas_size, freq, theta_context)
            context = Image.fromarray(np.uint8(context * 255))
        else:
            context = self.create_canvas()
        test = generate_grating(self.canvas_size, freq, theta_test)
        test = Image.fromarray(np.uint8(test * 255))
        mask = Image.new("L", test.size, 0)

        draw = ImageDraw.Draw(mask)
        center_test = np.array(center_test) * self.canvas_size
        draw.ellipse(
            (
                center_test[0] - radius,
                center_test[1] - radius,
                center_test[0] + radius,
                center_test[1] + radius,
            ),
            fill=255,
        )

        context.paste(test, mask=mask)
        return apply_antialiasing(context) if self.antialiasing else context


DEFAULTS.update(
    {
        "num_only_center_samples": 100,
        "num_center_context_samples": 100,
        "output_folder": "data/low_level_vision/tilt_illusion",
    }
)


def generate_all(
    num_only_center_samples=DEFAULTS["num_only_center_samples"],
    num_center_context_samples=DEFAULTS["num_center_context_samples"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    def get_random_values():
        size_scale = np.random.uniform(0.1, 0.6)
        radius = canvas_size[0] // 2 * size_scale
        center = (
            np.random.uniform(radius, canvas_size[0] - radius) // canvas_size[0],
            np.random.uniform(radius, canvas_size[1] - radius) // canvas_size[1],
        )
        freq = random.randint(5, 20)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        return theta, radius, center, freq

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
        for i in ["only_center", "center_context"]
    ]

    ds = DrawTiltIllusion(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "Background",
                "ThetaCenter",
                "Radius",
                "Frequency",
                "ThetaContext",
                "IterNum",
            ]
        )
        for i in tqdm(range(num_only_center_samples)):
            theta_center, radius, _, freq = get_random_values()
            path = Path("only_center") / f"{-theta_center:.3f}__0_{i}.png"
            img = ds.generate_illusion(theta_center, radius, (0.5, 0.5), freq)
            img.save(str(output_folder / path))
            writer.writerow(
                [path, "only_center", ds.background, theta_center, radius, freq, "", i]
            )

        all_thetas = np.linspace(-np.pi / 2, np.pi / 2, num_center_context_samples)
        for i, theta_context in enumerate(tqdm(all_thetas)):
            _, radius, _, freq = get_random_values()
            img = ds.generate_illusion(0, radius, (0.5, 0.5), freq, theta_context)
            path = Path("center_context") / f"0__{theta_context:.3f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow(
                [
                    path,
                    "center_context",
                    ds.background,
                    0,
                    radius,
                    freq,
                    theta_context,
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

    parser.add_argument(
        "--num_only_center_samples",
        "-ncenter",
        help="Number of samples with only the center gabor patch",
        default=DEFAULTS["num_only_center_samples"],
        type=int,
    )
    parser.add_argument(
        "--num_center_context_samples",
        "-ncontext",
        default=DEFAULTS["num_center_context_samples"],
        help="Number of samples for center and context gabor patches",
        type=int,
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
