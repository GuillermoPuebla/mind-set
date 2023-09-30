import os
import toml
from src.utils.decoder.examples.regression.jastrow_illusion.jastrow_utils import (
    adjust_csv,
)

from src.utils.decoder.train import decoder_train
import pathlib


dataset_folder = pathlib.Path("datasets_examples") / "jastrow_illusion"

# We create a small Jastrow_Illusion dataset just for this example.
from src.generate_datasets.low_level_vision.jastrow_illusion.generate_dataset import (
    generate_all as Jastrow_generate,
)

if not pathlib.Path(dataset_folder).exists():
    Jastrow_generate(
        output_folder=dataset_folder,
        num_samples_random=100,
        num_samples_aligned=100,
        num_samples_illusory=100,
    )

# The regression task consists of predicting the difference between the top and bottom shape. This is not provided in the annotation file but it's easy to add it ourselves:
new_ann_path = dataset_folder / "annotation_w_diff.csv"

if not new_ann_path.exists():
    difference_df = adjust_csv(
        dataset_folder / "annotation.csv",
        "SizeTop",
        "SizeBottom",
        "DiffSizeTopBottom",
    )
    difference_df.to_csv(new_ann_path, index=False)

# We then perform the training through the `toml` config file.
with open(os.path.dirname(__file__) + "/jastrow_train.toml", "r") as f:
    toml_config = toml.load(f)
decoder_train(**toml_config)

##
