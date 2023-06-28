import argparse
import csv

from src.gestalt.CSE_CIE_dots.utils import DrawCSE_CIEdots
import pathlib

from src.utils.misc import add_general_args, delete_and_recreate_path


def generate_all(output_folder, canvas_size, background, antialiasing, num_samples):
    dr = DrawCSE_CIEdots(
        background=background,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=10,
    )

    all_types = ["single", "proximity", "orientation", "linearity"]

    output_folder = (
        pathlib.Path("data") / "gestalt" / "CSE_CIE_dots"
        if output_folder is None
        else output_folder
    )

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
        default=100,
        help="Each `sample` corresponds to an entire set of pair of stimuli, for each condition.",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
