import os
import toml

from src.utils.decoder.eval import decoder_evaluate
import pathlib

from src.utils.decoder.examples.regression.jastrow_illusion.jastrow_utils import (
    adjust_csv,
)


annotation_file = (
    pathlib.Path("datasets")
    / "low_level_vision"
    / "jastrow_illusion"
    / "annotation.csv"
)

new_ann_path = annotation_file.parent / "annotation_w_diff.csv"
if not new_ann_path.exists():
    difference_df = adjust_csv(
        annotation_file,
        "SizeTop",
        "SizeBottom",
        "DiffSizeTopBottom",
    )
    difference_df.to_csv(new_ann_path, index=False)


with open(os.path.dirname(__file__) + "/jastrow_regression_train.toml", "r") as f:
    toml_config = toml.load(f)
decoder_evaluate(**toml_config)
