from src.utils.compute_distance.run import compute_distance

folder = "data/obj_representation/leek_reppa_arguin_2005/"
pretraining = ["ImageNet", "vanilla"]
for p in pretraining:
    compute_distance(
        input_paths=dict(
            base_name=f"{folder}/base",
            folder=folder,
        ),
        options=dict(pretraining=p),
        saving_folders=dict(
            result_folder=f"results/distance_similarity/obj_representation/leek_reppa_arguin_2005/{p}"
        ),
        transformation=dict(repetitions=50),
    )
