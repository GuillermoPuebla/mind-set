from src.utils.compute_distance.run import compute_distance

folder = "obj_representation/baker_2018/grid_degree45/"
sizes = [8, 16]
pretraining = ["ImageNet", "vanilla"]
for g in sizes:
    for p in pretraining:
        compute_distance(
            input_paths=dict(
                base_name=f"data/{folder}/gsize{g}/del",
                folder=f"data/{folder}/gsize{g}/",
            ),
            options=dict(pretraining=p),
            saving_folders=dict(
                result_folder=f"results/distance_similarity/{folder}/gsize{g}/{p}/"
            ),
            transformation=dict(repetitions=50, matching_transform=False),
            folder_vs_folder=dict(match_mode="all"),
        )
