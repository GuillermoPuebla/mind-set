import argparse

from src.gestalt.similarity_CSE_CIE_dots.utils import DrawCSE_CIEdots
import pathlib

from src.utils.misc import add_general_args


def generate_all(output_folder, canvas_size, background, antialiasing, num_samples):
    dr = DrawCSE_CIEdots(
        background=background,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=10,
    )

    all_types = ["single", "proximity", "orientation", "linearity"]

    output = (
        pathlib.Path("data/gestalt/similarity_CSE_CIE_dots/")
        if output_folder is None
        else output_folder
    )

    [
        [(output / i / j).mkdir(exist_ok=True, parents=True) for i in all_types]
        for j in ["a", "b"]
    ]

    for i in range(num_samples):
        all = dr.get_all_sets()[0]
        for t in all_types:
            all[t][0].save(output / t / "a" / f"{i}.png")
            all[t][1].save(output / t / "b" / f"{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--num_samples",
        default=1000,
        help="Each `sample` corresponds to an entire set of pair of stimuli, for each condition.",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
