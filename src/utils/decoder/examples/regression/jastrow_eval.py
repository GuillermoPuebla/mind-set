import os
import toml

from src.utils.decoder.eval import decoder_evaluate

with open(os.path.dirname(__file__) + "/jastrow_regression_test.toml", "r") as f:
    toml_config = toml.load(f)
decoder_evaluate(**toml_config)
