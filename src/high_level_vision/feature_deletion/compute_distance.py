from src.utils.compute_distance.run import compute_distance

folder = "data/high_level_vision/feature_deletion/"
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        input_paths=dict(
            base_name=f"{folder}/del1",
            folder=folder,
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/distance_similarity/high_level_vision/feature_deletion/{p}"
        ),
        transformation=dict(
            repetitions=50,
            matching_transform=False,
            copy_on_bigger_canvas=True,
            canvas_to_image_ratio=2,
        ),
    )
