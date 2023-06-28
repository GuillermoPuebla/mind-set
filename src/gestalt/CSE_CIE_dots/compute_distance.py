from src.utils.compute_distance.run import compute_distance
from pathlib import Path

"""
Here we compute the distance for each condition separately. Then during the analysis stage, we need to compare the distance across conditions. 
"""
conditions = ["single", "proximity", "linearity", "orientation"]
data_folder = Path("data") / "gestalt" / "CSE_CIE_dots"


pretraining = ["ImageNet", "vanilla"]
for condition in conditions:
    for p in pretraining:
        compute_distance(
            input_paths=dict(
                base_name=data_folder / condition / "a",
                folder=data_folder / condition / "b",
            ),
            options=dict(pretraining=p),
            saving_folders=dict(
                result_folder=Path(
                    "results"
                    / "distance_similarity"
                    / "gestalt"
                    / "CSE_CIE_dots"
                    / condition
                ),
            ),
            transformation=dict(repetitions=50, matching_transform=False),
            folder_vs_folder=dict(match_mode="all"),
        )
