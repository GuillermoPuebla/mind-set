import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import sty
from PIL import ImageDraw

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
    DEFAULTS,
)


def draw_arrow(draw, pos, theta, angle_arrow, arrow_length, width, color):
    x, y = pos
    arrow_theta1 = theta - angle_arrow
    arrow_theta2 = theta + angle_arrow

    arrow_end_x1 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta1)))
    arrow_end_y1 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta1)))
    arrow_end_x2 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta2)))
    arrow_end_y2 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta2)))

    # Draw the arrow lines
    draw.line([(x, y), (arrow_end_x1, arrow_end_y1)], fill=color, width=width)
    draw.line([(x, y), (arrow_end_x2, arrow_end_y2)], fill=color, width=width)


class DrawMullerLyer(DrawStimuli):
    def generate_illusion(
        self,
        line_position_rel,
        line_length,
        arrow_angle,
        arrow_cap_angle,
        arrow_length,
        type,
    ):
        get_arrow_rnd_pos = lambda: (
            random.randint(arrow_length, self.canvas_size[0] - arrow_length),
            random.randint(arrow_length, self.canvas_size[1] - arrow_length),
        )

        img = self.create_canvas()
        d = ImageDraw.Draw(img)
        line_position = tuple(
            (np.array(line_position_rel) * self.canvas_size).astype(int)
        )
        if type == "scrambled":
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle + 180,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            d.line(
                (
                    np.round(line_position[0] - line_length // 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length // 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )
        else:
            d.line(
                (
                    np.round(line_position[0] - line_length / 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length / 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )

            draw_arrow(
                d,
                (line_position[0] - line_length // 2, line_position[1]),
                theta=(180 if type == "test_outward" else 0),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                (line_position[0] + line_length // 2, line_position[1]),
                theta=(0 if type == "test_outward" else 180),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
        return apply_antialiasing(img) if self.antialiasing else img


def generate_all(
    num_scrambled_samples,
    num_illusory_samples,
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    output_folder = (
        Path("data") / "low_level_vision" / "muller_lyer_illusion"
        if output_folder is None
        else Path(output_folder)
    )
    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return output_folder

    delete_and_recreate_path(output_folder)
    conditions = ["scrambled", "mulllyer_inward", "mulllyer_outward"]
    [(output_folder / i).mkdir(exist_ok=True, parents=True) for i in conditions]

    ds = DrawMullerLyer(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=1,
    )

    def get_random_params():
        line_length = random.randint(
            int(canvas_size[0] * 0.25), int(canvas_size[0] * 0.67)
        )
        arrow_length = random.randint(
            int(canvas_size[0] * 0.07), int(canvas_size[0] * 0.134)
        )
        line_position = tuple(
            np.array(
                [
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[0] - arrow_length - line_length // 2,
                    ),
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[1] - arrow_length - line_length // 2,
                    ),
                ],
            )
            / canvas_size
        )
        cap_arrows_angle = random.randint(
            int(canvas_size[0] * 0.045), int(canvas_size[1] * 0.2)
        )
        angle_arrow = random.randint(0, 360)
        return line_length, line_position, arrow_length, cap_arrows_angle, angle_arrow

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "Background",
                "LineLength",
                "LinePosition",
                "ArrowLength",
                "CapArrowAngle",
                "ArrowAngle",
                "IterNum",
            ]
        )
        for c in conditions:
            num_samples = (
                num_scrambled_samples if c == "scrambled" else num_illusory_samples
            )
            for i in range(num_samples):
                (
                    line_length,
                    line_position,
                    arrow_length,
                    cap_arrow_angle,
                    arrow_angle,
                ) = get_random_params()

                img = ds.generate_illusion(
                    line_position_rel=line_position,
                    line_length=line_length,
                    arrow_angle=arrow_angle,
                    arrow_cap_angle=cap_arrow_angle,
                    arrow_length=arrow_length,
                    type="scrambled",
                )
                path = Path(c) / f"{line_length}__{i}.png"
                img.save(str(output_folder / path))
                writer.writerow(
                    [
                        path,
                        c,
                        background,
                        line_length,
                        line_position,
                        arrow_length,
                        cap_arrow_angle,
                        arrow_angle,
                        i,
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    # add_training_args(parser)
    parser.add_argument(
        "--num_scrambled_samples",
        "-nscrambled",
        default=100,
        type=int,
    )
    parser.add_argument("--num_illusory_samples", "-nill", default=100, type=int)
    parser.set_defaults(antialiasing=False)
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
