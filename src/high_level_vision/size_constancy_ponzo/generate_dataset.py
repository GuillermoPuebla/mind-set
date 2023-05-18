import argparse
import shutil
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import random

from src.utils.misc import apply_antialiasing


def get_random_start_end_ponts(canvas_size):
    start_point = (
        random.randint(0, canvas_size[0]),
        random.randint(0, canvas_size[1]),
    )
    end_point = (
        random.randint(0, canvas_size[0]),
        random.randint(0, canvas_size[1]),
    )
    return start_point, end_point


def generate_train_image(
    canvas_size, num_black_lines, colored_line_always_horizontal=False, antialias=True
):
    img = Image.new("RGB", canvas_size, color="black")

    d = ImageDraw.Draw(img)
    for i in range(num_black_lines):
        start_point, end_point = get_random_start_end_ponts(canvas_size)
        d.line([start_point, end_point], fill="white", width=2)

    if colored_line_always_horizontal:
        margin = canvas_size[0] * 0.11
        v_position_red = random.randint(int(margin), int(canvas_size[1] - margin))
        v_position_blue = random.randint(int(margin), int(canvas_size[1] - margin))
        red_length = random.randint(canvas_size[0] // 10, canvas_size[0] // 2)
        blue_length = random.randint(canvas_size[0] // 10, canvas_size[0] // 2)
        red_sp = (canvas_size[0] // 2 - red_length // 2, v_position_red)
        red_ep = (canvas_size[0] // 2 + red_length // 2, v_position_red)
        blue_sp = (canvas_size[0] // 2 - blue_length // 2, v_position_blue)
        blue_ep = (canvas_size[0] // 2 + blue_length // 2, v_position_blue)

    else:
        red_sp, red_ep = get_random_start_end_ponts(canvas_size)
        blue_st, blue_ep = get_random_start_end_ponts(canvas_size)

    d.line([red_sp, red_ep], fill="red", width=2)
    red_length = np.linalg.norm(np.array(red_ep) - np.array(red_sp))
    d.line([blue_sp, blue_ep], fill="blue", width=2)
    blue_length = np.linalg.norm(np.array(blue_ep) - np.array(blue_sp))

    max_length = canvas_size[0] // 2 - canvas_size[0] // 10
    label = red_length - blue_length
    norm_label = label / max_length
    if antialias:
        img = img.resize(tuple(np.array(canvas_size) * 2)).resize(
            canvas_size, resample=Image.Resampling.LANCZOS
        )
    return img, norm_label


def compute_x(y, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x = (y - b) / m
    return x


def generate_test_image(
    canvas_size, num_horizontal_lines, same_length=False, antialias=False
):
    img = Image.new("RGB", canvas_size, color="black")
    d = ImageDraw.Draw(img)
    margin = canvas_size[0] * 0.11

    ## draw the diagonal
    vertical_distance_from_center = random.randint(
        canvas_size[0] * 0.11, canvas_size[0] * 0.33
    )
    x = canvas_size[0] // 2 - vertical_distance_from_center
    length = canvas_size[0] - margin * 2
    slope = random.random() * 2 + 3.2  # random.random()
    x2 = x + length // 2 * math.cos(math.atan(-slope))
    y2 = canvas_size[1] // 2 + length // 2 * math.sin(math.atan(-slope))
    x1 = x - length // 2 * math.cos(math.atan(-slope))
    y1 = canvas_size[1] // 2 - length // 2 * math.sin(math.atan(-slope))
    d1_line_start = (x1, y1)
    d1_line_end = (x2, y2)
    d.line([d1_line_start, d1_line_end], fill="white", width=2)

    x = canvas_size[0] // 2 + vertical_distance_from_center
    x2 = x + length // 2 * math.cos(math.atan(slope))
    y2 = canvas_size[1] // 2 + length // 2 * math.sin(math.atan(slope))
    x1 = x - length // 2 * math.cos(math.atan(slope))
    y1 = canvas_size[1] // 2 - length // 2 * math.sin(math.atan(slope))
    d2_line_start = (x1, y1)
    d2_line_end = (x2, y2)

    d.line([d2_line_start, d2_line_end], fill="white", width=2)

    if num_horizontal_lines > 0:
        vertical_ranges = [
            int(i * (canvas_size[1] - margin * 2) // (num_horizontal_lines) + margin)
            for i in range(num_horizontal_lines + 1)
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
    v_position_up = random.randint(int(margin), canvas_size[1] // 2)
    v_position_down = random.randint(canvas_size[1] // 2, int(canvas_size[1] - margin))
    pos = np.array([v_position_down, v_position_up])
    red_length = random.randint(canvas_size[0] // 10, canvas_size[0] // 2)
    blue_length = (
        red_length
        if same_length
        else random.randint(canvas_size[0] // 10, canvas_size[0] // 2)
    )
    np.random.shuffle(pos)

    upper_line_color = "red" if pos[0] < pos[1] else "blue"
    d.line(
        (
            (canvas_size[0] // 2 - red_length // 2, pos[0]),
            (canvas_size[0] // 2 + red_length // 2, pos[0]),
        ),
        fill="red",
        width=2,
    )

    d.line(
        (
            (canvas_size[0] // 2 - blue_length // 2, pos[1]),
            (canvas_size[0] // 2 + blue_length // 2, pos[1]),
        ),
        fill="blue",
        width=2,
    )

    max_length = canvas_size[0] // 2 - canvas_size[0] // 10
    label = red_length - blue_length
    norm_label = label / max_length

    img = apply_antialiasing() if antialias else img

    return img, norm_label, upper_line_color


def generate_all(
    canvas_size,
    num_training_samples,
    num_testing_samples,
    num_rail_lines,
    train_with_rnd_col_lines,
    antialiasing,
):
    output_folder = Path("data/high_level_vision/size_constancy_ponzo/")

    [
        shutil.rmtree(output_folder / i) if (output_folder / i).exists() else None
        for i in ["train", "test_same_length", "test_diff_length"]
    ]

    [
        (output_folder / i).mkdir(exist_ok=True, parents=True)
        for i in ["train", "test_same_length", "test_diff_length"]
    ]

    for i in range(num_training_samples):
        img, norm_label = generate_train_image(
            canvas_size,
            num_rail_lines,
            colored_line_always_horizontal=not train_with_rnd_col_lines,
            antialias=antialiasing,
        )
        img.save(output_folder / "train" / f"{norm_label:.3f}_{i}.png")

    for i in range(num_testing_samples):
        img, norm_label, upper_line_color = generate_test_image(
            canvas_size, num_rail_lines, same_length=True, antialias=antialiasing
        )
        img.save(
            output_folder
            / "test_same_length"
            / f"{norm_label:.3f}_{upper_line_color}_{i}.png"
        )

        img, norm_label, upper_line_color = generate_test_image(
            canvas_size, num_rail_lines, same_length=False, antialias=antialiasing
        )
        img.save(
            output_folder
            / "test_diff_length"
            / f"{norm_label:.3f}_{upper_line_color}_{i}.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default="224x224",
        help="A string in the format NxM specifying the size of the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")]),
    )
    parser.add_argument(
        "--num_training_samples",
        "-ntrain",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--num_testing_samples",
        "-ntest",
        default=5000,
        help="Two testing folders will be generated, each one with ntest samples. Both will have the `railway` configuration, but in one case the red and blue lines will have different length, in the other one will have the same length. Only this latter case constitute the illusion of interest, the other folder can be used for testing",
        type=int,
    )
    parser.add_argument(
        "--num_rail_lines",
        "-nrl",
        default=5,
        help="This refers to the number of horizontal lines (excluding the red and blue lines) in the proper illusion stimuli. During training, we generate dataset matching the total number of lines, so that this parameter will affect both test and train stimuli. Notice that, since in the minimal illusion, two oblique lines are always present, similarly in the train stimuli there are always two lines, to which we add a number of lines specified by this parameter",
        type=int,
    )

    parser.add_argument(
        "--train_with_rnd_col_lines",
        "-trc",
        help="Specify whether the red and blue lines in  training dataset should be randomly placed, or should be horizontal like in the testing (illusion) condition",
        action="store_true",
    )

    parser.add_argument(
        "--no_antialiasing",
        "-nantial",
        dest="antialiasing",
        help="Specify whether we want to disable antialiasing",
        action="store_false",
        default=True,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
