"""
This file will perform all the operations that needs to be done when all code is written: 
- generate the toml configs (for both full and lite version), 
- generate the datasets (full and lite), 
- and publish the datasets on kaggle.
"""

import glob
from src.generate_datasets_from_toml import generate_datasets_from_toml_file
import src.utils.generate_default_pars_toml_file
from src.utils.misc import modify_toml
from src.utils.publish_kaggle import publish


generate_tomls = True
generate_datasets = True
publish_kaggle = True

toml_all_full = "generate_all_datasets.toml"
toml_all_lite = "generate_all_datasets_lite.toml"

if generate_tomls:
    # src.utils.generate_default_pars_toml_file.create_config(toml_all_full)
    src.utils.generate_default_pars_toml_file.generate_lite(
        toml_all_full, toml_all_lite
    )


if generate_datasets:
    generate_datasets_from_toml_file(toml_all_full)
    generate_datasets_from_toml_file(toml_all_lite)

if publish_kaggle:
    publish(data_tye="full")
    publish(data_tye="lite")
