import argparse
import csv

import sty

from src.datasets_generation.gestalt.CSE_CIE_dots.utils import DrawCSE_CIEdots
import pathlib

from src.utils.misc import add_general_args, delete_and_recreate_path, DEFAULTS

DEFAULTS["num_samples"] = 100


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    num_samples=DEFAULTS["num_samples"],
    regenerate=DEFAULTS["regenerate"],
):
    dr = DrawCSE_CIEdots(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=10,
    )

    all_types = ["single", "proximity", "orientation", "linearity"]

    output_folder = (
        pathlib.Path("data") / "gestalt" / "CSE_CIE_dots"
        if output_folder is None
        else pathlib.Path(output_folder)
    )

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return output_folder

    delete_and_recreate_path(output_folder)

    [
        [(output_folder / i / j).mkdir(exist_ok=True, parents=True) for i in all_types]
        for j in ["a", "b"]
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "PairA/B", "IterNum"])
        for i in range(num_samples):
            all = dr.get_all_sets()[0]
            for t in all_types:
                for ip, pair in enumerate(["a", "b"]):
                    path = pathlib.Path(t) / pair / f"{i}.png"
                    all[t][ip].save(output_folder / path)
                    writer.writerow([path, t, pair, i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=DEFAULTS["num_samples"],
        help="Each `sample` corresponds to an entire set of pair of shape_based_image_generation, for each condition.",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
