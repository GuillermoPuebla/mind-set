from src.utils.compute_distance.run import compute_distance

folder = "data/obj_representation/biedermann_1985_1987/"
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        input_paths=dict(
            base_name=f"{folder}/del1",
            folder=folder,
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/distance_similarity/obj_representation/biedermann_1985_1987/{p}"
        ),
        transformation=dict(
            repetitions=50,
            matching_transform=False,
            copy_on_bigger_canvas=True,
            canvas_to_image_ratio=2,
        ),
    )
