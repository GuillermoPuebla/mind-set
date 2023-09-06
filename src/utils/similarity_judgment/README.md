# Similarity Judgment Analysis

## Method Overview

This method allows the analysis of the internal activation of a network when fed with different images. Images are always compared in pairs. Each image is fed individually, the internal activation for the requested layers are extracted, and then the two sets of activations are compared using either the Euclidean Distance or the Cosine Similarity (other comparisons technique might be added in the future).

## Code Overview

### Main Function

The main function is `compute_distance` in `src/utils/similarity_judgment/run.py`. This function requires a set of parameters that can be provided through a `toml` file (see the example folder and specifically [`examples/CSE_CIE_dots_config.toml`](./examples/CSE_CIE_dots_config.toml)). The `toml` file will refer to an `annotation.csv` file, which is automatically created when a dataset is generated. For example, the example toml file refers to the [`datasets/gestalt/CSE_CIE_dots/annotation.csv`](../../../datasets/gestalt/CSE_CIE_dots/annotation.csv) file.

### Configuration Parameters

The `toml` file for the similarity judgment method can have many parameters. You can see _all_ parameters in the `default_distance_config.toml`. When you run `compute_distance` providing your own toml file, any parameter that is _not_ included in your `toml` file will default to the parameter specified in the [`default_distance_config.toml`](default_distance_config.toml). In this way, your config file can be kept relatively short, and only contain the relevant parameters. Note that the `compute_distance` will create, in the `results_folder`, a `toml` file containing _all_ the parameters used in the analysis, including the default ones.

An explanation of each single parameter can be found in the [`default_distance_config.toml`](default_distance_config.toml).

## Create a Report

To generate a succinct report based on the data collated in the `dataframe.csv` file, execute the command `src/utils/generate_report.py --results_csv_path path/to/dataframe.csv`. If the `--results_csv_path` parameter is omitted, the script defaults to identifying all `*/dataframe.csv` files in the `results/similarity_judgments` folder, generating individual reports for each. These reports comprise a set of images and a `.ipynb` file, conveniently stored in the same directory as the respective `dataframe.csv` file.
