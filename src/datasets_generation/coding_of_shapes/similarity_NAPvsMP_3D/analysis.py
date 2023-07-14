from pathlib import Path

from src.utils.compute_distance.analysis import (
    run_all_layers_analysis,
    run_standard_analysis_one_layer,
)
%load_ext autoreload
%autoreload 2

dataset_name = Path("coding_of_shapes") / "NAPvsMP_3D"

pretraining = "vanilla"
run_standard_analysis_one_layer(dataset_name, pretraining, -1)
run_all_layers_analysis(dataset_name, pretraining)

pretraining = "ImageNet"
run_standard_analysis_one_layer(dataset_name, pretraining, -1)
run_all_layers_analysis(dataset_name, pretraining, ylim=[0, 50], xlim=[130, 157])
