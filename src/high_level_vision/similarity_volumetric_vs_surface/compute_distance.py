from src.utils.compute_distance.run import compute_distance

folder = "data/high_level_vision/similarity_volumetric_vs_surface/"
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        input_paths=dict(
            base_name=f"{folder}/base",
            folder=folder,
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/distance_similarity/high_level_vision/similarity_volumetric_vs_surface/{p}"
        ),
        transformation=dict(repetitions=50),
    )
