import os
import pandas as pd
import toml
from src.utils.decoder.train import run_train
import pathlib


def adjust_csv(csv_file_path: str, col1: str, col2: str, new_col_name: str):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(
            f"One or both of the specified columns ({col1}, {col2}) do not exist in the DataFrame"
        )

    # Compute the difference between the two columns
    df[new_col_name] = df[col1] - df[col2]

    return df


annotation_file = (
    pathlib.Path("data") / "low_level_vision" / "jastrow_illusion" / "annotation.csv"
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

with open(os.path.dirname(__file__) + "/train.toml", "r") as f:
    toml_config = toml.load(f)
run_train(**toml_config)

##
