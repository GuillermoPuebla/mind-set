import os
import pandas as pd
import toml
import inspect

from src.utils.dataset_utils import ImageNetClasses
from src.utils.imagenet_classification.eval import classification_evaluate
import pathlib


dataset_folder = pathlib.Path("datasets_examples") / "linedrawings"

# We create a small Jastrow_Illusion dataset just for this example.
from src.generate_datasets.shape_and_object_recognition.linedrawings.generate_dataset import (
    generate_all as linedrawings_generate,
)

if not pathlib.Path(dataset_folder).exists():
    linedrawings_generate(
        output_folder=dataset_folder,
    )

# We need to add the ImageNetClassIndex column, based on the "Class" column.
add_class = ImageNetClasses()
add_class.add_to_annotation_file_path(
    str(dataset_folder / "annotation.csv"),
    "Class",
    str(dataset_folder / "annotation_w_imagenet_idxs.csv"),
)

with open(os.path.dirname(__file__) + "/linedrawings_config.toml", "r") as f:
    toml_config = toml.load(f)
classification_evaluate(**toml_config)

##
