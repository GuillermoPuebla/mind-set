import argparse

from src.gestalt.CSE_CIE_dots.utils import DrawShape
import pathlib

### Generate Stimuli for Experiment 1
background = "random"
dr = DrawShape(background="black", img_size=(224, 224), width=10)

all_types = ["single", "proximity", "orientation", "linearity"]

output = pathlib.Path("data/gestalt/CSE_CIE_dots/")
[
    [(output / i / j).mkdir(exist_ok=True, parents=True) for i in all_types]
    for j in ["a", "b"]
]


def generate(num_samples):
    for i in range(num_samples):
        all = dr.get_all_sets()[0]
        for t in all_types:
            all[t][0].save(output / t / "a" / f"{i}.png")
            all[t][1].save(output / t / "b" / f"{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        default=1000,
        help="Each `sample` corresponds to an entire set of pair of stimuli, for each condition.",
    )
    #
    args = parser.parse_known_args()[0]
    run(**args.__dict__)
