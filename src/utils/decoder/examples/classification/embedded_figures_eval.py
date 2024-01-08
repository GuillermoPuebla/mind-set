import os
import pandas as pd
import toml
import inspect

from src.utils.decoder.eval import decoder_evaluate
import pathlib


dataset_folder = pathlib.Path("datasets_examples") / "embedded_figures"

# We create a small Jastrow_Illusion dataset just for this example.
from src.generate_datasets.shape_and_object_recognition.embedded_figures.generate_dataset import (
    generate_all as embedded_figures_generate,
)

if not pathlib.Path(dataset_folder).exists():
    embedded_figures_generate(
        output_folder=dataset_folder,
        num_samples=1000,
    )


with open(os.path.dirname(__file__) + "/embedded_figures_eval.toml", "r") as f:
    toml_config = toml.load(f)
decoder_evaluate(**toml_config)

##
