import pathlib
from src.generate_dataset import (
    generate_all,
)
from src.utils.compute_distance.run import compute_distance

dataset_folder = generate_all(regenerate=False)

pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        basic_info=dict(
            annotation_file_path=pathlib.Path(dataset_folder) / "annotation.csv",
            match_factors=["ObjectNum"],
            factor_variable="Type",
            reference_level="reference",
            filter_factor_level={},
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/coding_of_shapes/volumetric_vs_surface/{p}"
        ),
        transformation=dict(repetitions=20),
    )
