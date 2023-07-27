import argparse
import csv
import glob
import os
import shutil

import sty
from tqdm import tqdm
import pathlib
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from src.datasets_generation.gestalt.embedded_figures.utils import DrawEmbeddedFigures
from src.utils.compute_distance.misc import my_affine, get_new_affine_values
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()

polys = [
    [
        (57.24191632273875, 56.91324204660534),
        (67.83138439623637, 82.164285681352),
        (32.38511628951794, 100.0),
        (30.804578657842352, 73.57800097038611),
        (39.790659001314935, 49.086636164923235),
        (0.0, 13.7323319073141),
        (94.72413166442718, 12.682935522851094),
        (57.24191632273875, 56.91324204660534),
    ],
    [
        (81.48562751057788, 66.09981880428393),
        (37.92965499595768, 77.0744200152708),
        (0.0, 68.97209639097005),
        (44.9389941502286, 34.540151429883004),
        (53.987122366845064, 49.564278276404),
        (84.56537465354238, 18.919456992657267),
        (100.0, 18.699487501673413),
        (81.48562751057788, 66.09981880428393),
    ],
    [
        (35.97327747005214, 45.791246316377155),
        (11.785178465160033, 100.0),
        (27.099520693118553, 42.4229653007916),
        (14.996849174804217, 38.15001140882252),
        (0.0, 14.776880790143396),
        (72.18476581016824, 0.4054647678679052),
        (88.90514143051871, 30.360919479254665),
        (35.97327747005214, 45.791246316377155),
    ],
    [
        (71.10439849304493, 58.11478725863726),
        (61.9937434893466, 60.73142522572432),
        (92.85669179071442, 92.82864165606136),
        (88.72099545300055, 99.99999999999999),
        (0.0, 39.73040200323956),
        (21.971533124959556, 37.83178086979938),
        (24.413419434667112, 10.00396765284736),
        (71.10439849304493, 58.11478725863726),
    ],
    [
        (99.99999999999999, 68.62298822991318),
        (1.0462848992740792, 74.25606525894862),
        (39.38222527207329, 41.822553930272285),
        (28.55589809787002, 40.16660427664721),
        (13.075399744528019, 0.0),
        (52.12530258921337, 38.638493749406464),
        (79.2845572364489, 40.26425593476456),
        (99.99999999999999, 68.62298822991318),
    ],
]

DEFAULTS.update(
    {
        "num_samples_polygons": 100,
        "num_samples_embedded_polygons": 100,
        "shape_size": 45,
        "debug_mode": False,
        "output_folder": "data/gestalt/embedded_figures",
    }
)


def generate_all(
    num_samples_polygons=DEFAULTS["num_samples_polygons"],
    num_samples_embedded_polygons=DEFAULTS["num_samples_embedded_polygons"],
    shape_size=DEFAULTS["shape_size"],
    debug_mode=DEFAULTS["debug_mode"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and `regenerate` flag if false. Finished"
            + sty.rs.fg
        )
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    shapes = list(((n, p) for n, p in enumerate(polys)))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "Background", "IterNum"])
        ds = DrawEmbeddedFigures(
            shape_size=shape_size,
            canvas_size=canvas_size,
            background=background_color,
            antialiasing=antialiasing,
        )
        for cond in tqdm(["polygons", "embedded_polygons"]):
            N = (
                num_samples_polygons
                if cond == "train"
                else num_samples_embedded_polygons
            )

            for s in tqdm(shapes, leave=False):
                shape_name, shape_points = str(s[0]), s[1]
                img_path = pathlib.Path(cond) / shape_name
                class_folder = output_folder / img_path
                class_folder.mkdir(parents=True, exist_ok=True)
                for i in tqdm(range(N)):
                    if cond == "polygons":
                        main_image = ds.draw_shape(
                            shape_points,
                            extend_lines=False,
                            num_shift_lines=0,
                            num_rnd_lines=0,
                            debug=debug_mode,
                        )

                    else:
                        main_image = ds.draw_shape(
                            shape_points,
                            extend_lines=True,
                            num_shift_lines=10,
                            num_rnd_lines=10,
                            debug=debug_mode,
                        )
                    af = get_new_affine_values("r[-180, 180]t[-0.1, 0.1]s[0.5, 2]")
                    img = my_affine(
                        main_image,
                        translate=af["tr"],
                        angle=af["rt"],
                        scale=af["sc"],
                        shear=af["sh"],
                        interpolation=InterpolationMode.BILINEAR,
                        fill=ds.background,
                    )
                    img = transforms.CenterCrop((canvas_size[1], canvas_size[0]))(img)
                    img.save(class_folder / f"{i}.png")
                    writer.writerow([img_path, cond, ds.background, i])
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument("--debug_mode", "-debug", action="store_true")
    parser.add_argument(
        "--num_samples_polygons",
        "-nsp",
        help="The number of samples to generate for each (non-embedded) polygons",
        default=DEFAULTS["num_samples_polygons"],
        type=int,
    )

    parser.add_argument(
        "--num_samples_embedded_polygons",
        "-nsep",
        help="The number of samples to generate for each (embedded) polygons",
        default=DEFAULTS["num_samples_embedded_polygons"],
        type=int,
    )

    parser.add_argument(
        "--shape_size",
        "-shs",
        help="The shape of the embedded size (in pixels)",
        default=DEFAULTS["shape_size"],
        type=float,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)