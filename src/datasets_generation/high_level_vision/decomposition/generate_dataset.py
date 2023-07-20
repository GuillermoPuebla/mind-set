import argparse
import csv
import pathlib

import sty
from tqdm import tqdm

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import DEFAULTS, add_general_args, delete_and_recreate_path
from src.utils.shape_based_image_generation.modules.parent import ParentStimuli
from src.utils.shape_based_image_generation.modules.shapes import Shapes
from src.utils.shape_based_image_generation.utils.parallel import parallel_args
from itertools import product
import random
from pathlib import Path
import numpy as np
import uuid

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


class DrawDecomposition(DrawStimuli):
    def __init__(self, shape_size, shape_color, moving_distance, *args, **kwargs):
        self.shape_size = shape_size
        self.moving_distance = moving_distance
        self.shape_color = shape_color
        super().__init__(*args, **kwargs)

    def generate_canvas(
        self,
        shape_1_name,
        shape_2_name,
        split_type,
        cut_rotation,
        image_rotation,
        image_position,
    ):
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=4 if self.antialiasing else 1,
        )

        # create shapes -------------------------------------------
        shape_1 = Shapes(parent)
        shape_2 = Shapes(parent)
        getattr(shape_1, f"add_{shape_1_name}")(**{"size": self.shape_size})
        getattr(shape_2, f"add_{shape_2_name}")(**{"size": self.shape_size})
        shape_1.rotate(30)
        shape_2.rotate(30)

        shape_2.move_next_to(shape_1, "LEFT")

        if split_type == "no_split":
            shape_1.register()
            shape_2.register()

        elif split_type == "unnatural":
            piece_1, piece_2 = shape_2.cut(
                reference_point=(0.5, 0.5), angle_degrees=cut_rotation
            )
            index = np.argmax(
                [piece_1.get_distance_from(shape_1), piece_2.get_distance_from(shape_1)]
            )
            further_piece = [piece_1, piece_2][index]
            closer_piece = [piece_1, piece_2][1 - index]
            further_piece.move_apart_from(closer_piece, self.moving_distance)
            shape_1.register()
            piece_1.register()
            piece_2.register()

        elif split_type == "natural":
            shape_2.move_apart_from(shape_1, self.moving_distance)
            shape_1.register()
            shape_2.register()

        parent.binary_filter()
        parent.convert_color_to_color((255, 255, 255), self.shape_color)
        parent.move_to(image_position).rotate(image_rotation)
        self.create_canvas()  # dummy call to update the background for rnd-uniform mode
        parent.add_background(self.background)
        parent.shrink() if self.antialiasing else None
        return parent.canvas


def get_random_params():
    cut_rotation = random.uniform(0, 360)
    rotation = random.uniform(0, 360)
    position = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
    return cut_rotation, rotation, position


DEFAULTS.update(
    {
        "num_samples": 100,
        "moving_distance": 60,
        "shape_color": (255, 0, 0),
        "output_folder": "data/high_level_vision/decomposition",
    }
)


def generate_all(
    num_samples=DEFAULTS["num_samples"],
    moving_distance=DEFAULTS["moving_distance"],
    shape_color=DEFAULTS["shape_color"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    familiar_shapes = ["arc", "circle", "square", "rectangle", "polygon", "triangle"]
    unfamiliar_shapes = ["puddle"]

    combinations_familiar = list(product(familiar_shapes, familiar_shapes))
    combinations_familiar = [
        {
            "shape_1_name": shape_1_name,
            "shape_2_name": shape_2_name,
        }
        for shape_1_name, shape_2_name in combinations_familiar
    ]
    if len(combinations_familiar) < num_samples:
        print(
            f"The number of target_image_pairs_num is too small, it has to be at least larger than the number of combinations of familiar shapes. We set it to {len(combinations_familiar) + 1}"
        )
        num_samples = len(combinations_familiar) + 1

    if combinations_familiar:
        combinations_familiar *= int(num_samples / len(combinations_familiar))

    combinations_unfamiliar = list(product(unfamiliar_shapes, unfamiliar_shapes))
    combinations_unfamiliar = [
        {
            "shape_1_name": shape_1_name,
            "shape_2_name": shape_2_name,
        }
        for shape_1_name, shape_2_name in combinations_unfamiliar
    ]
    if len(combinations_unfamiliar) < num_samples:
        print(
            f"The number of target_image_pairs_num is too small, it has to be at least larger than the number of combinations of unfamiliar shapes. We set it to {len(combinations_unfamiliar) + 1}"
        )
        num_samples = len(combinations_unfamiliar) + 1
    if combinations_unfamiliar:
        combinations_unfamiliar *= int(num_samples / len(combinations_unfamiliar))

    shapes_types = {
        "familiar": combinations_familiar,
        "unfamiliar": combinations_unfamiliar,
    }
    shape_size = 0.05
    ds = DrawDecomposition(
        shape_size,
        shape_color,
        moving_distance,
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
    )

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    split_types = ["no_split", "unnatural", "natural"]
    [
        [
            (output_folder / name_comb / split_type).mkdir(exist_ok=True, parents=True)
            for split_type in split_types
        ]
        for name_comb in list(shapes_types.keys())
    ]
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Background",
                "ShapeType",
                "SplitType",
                "CutRotation",
                "Rotation",
                "Position",
                "SampleNum",
            ]
        )
        for name_comb, combs in tqdm(shapes_types.items()):
            for idx, c in enumerate(tqdm(combs, leave=False)):
                cut_rotation, rotation, position = get_random_params()
                for split_type in split_types:
                    img = ds.generate_canvas(
                        c["shape_1_name"],
                        c["shape_2_name"],
                        split_type=split_type,
                        cut_rotation=cut_rotation,
                        image_rotation=rotation,
                        image_position=position,
                    )
                    path = Path(name_comb) / split_type / f"{idx}.png"
                    img.save(output_folder / path)
                    writer.writerow(
                        [
                            path,
                            ds.background,
                            name_comb,
                            split_type,
                            cut_rotation,
                            rotation,
                            position,
                            idx,
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
        "--num_samples",
        "-ns",
        default=DEFAULTS["num_samples"],
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--moving_distance",
        "-mv",
        default=DEFAULTS["moving_distance"],
        type=int,
        help="Specify by how much each image is separated (same values for the whole dataset)",
    )
    parser.add_argument(
        "--shape_color",
        "-shpc",
        default=DEFAULTS["shape_color"],
        type=lambda x: tuple([int(i) for i in x.split("_")])
        if isinstance(x, str)
        else x,
        help="Specify the color of the shapes (same across the whole dataset). Specify in R_G_B format, e.g. 255_0_0 for red",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
