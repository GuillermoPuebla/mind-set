from pathlib import Path

from src.utils.compute_distance.analysis import (
    run_all_layers_analysis,
    run_standard_analysis_one_layer,
)

%load_ext autoreload
%autoreload 2

dataset_name = Path("coding_of_shapes") / "volumetric_vs_surface"
list_comparison_levels = ["small_vol", "small_surf"]
pretraining = "vanilla"
run_standard_analysis_one_layer(dataset_name, pretraining, -1, list_comparison_levels)
run_all_layers_analysis(dataset_name, pretraining, list_comparison_levels)

pretraining = "ImageNet"
run_standard_analysis_one_layer(dataset_name, pretraining, -1, list_comparison_levels)
run_all_layers_analysis(dataset_name, pretraining, list_comparison_levels, xlim=[0, 50])
run_all_layers_analysis(
    dataset_name, pretraining, list_comparison_levels, xlim=[50, 100], ylim=[0, 20]
)
run_all_layers_analysis(
    dataset_name, pretraining, list_comparison_levels, xlim=[100, 167], ylim=[0, 60]
)
