import json
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from src.utils.decoder.eval import decoder_evaluate


def analyse(json_file_path_blue_top: Path, json_file_path_red_top: Path):
    with open(json_file_path_blue_top, "r") as f:
        data_blue_top = json.load(f)

    with open(json_file_path_red_top, "r") as f:
        data_red_top = json.load(f)

    decoders = [i["decoder"] for i in data_blue_top]
    decoders = list(set(decoders))

    # RMSE after training for each decoder
    rmses = [
        0.06659,
        0.05981,
        0.03958,
        0.04501,
        0.12328,
        0.20841,
    ]
    rmses = iter(rmses)
    true_value = 0.0

    # plot the results on overlapping distributions curves
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, decoder in enumerate(decoders):
        data_blue_decoder = [
            i["prediction"] for i in data_blue_top if i["decoder"] == decoder
        ]
        data_red_decoder = [
            i["prediction"] for i in data_red_top if i["decoder"] == decoder
        ]
        sns.kdeplot(
            data_blue_decoder, ax=axs[i // 2, i % 2], label="blue", color="blue"
        )
        sns.kdeplot(data_red_decoder, ax=axs[i // 2, i % 2], label="red", color="red")
        # add the rmse line using 0 as the mean
        rmse_value = next(rmses)
        axs[i // 2, i % 2].axvline(x=rmse_value, color="green", linestyle="--")
        axs[i // 2, i % 2].axvline(x=-rmse_value, color="green", linestyle="--")
        axs[i // 2, i % 2].text(rmse_value, 0.5, "RMSE", rotation=90)
        axs[i // 2, i % 2].text(-rmse_value, 0.5, "RMSE", rotation=90)
        # add the true value line
        axs[i // 2, i % 2].axvline(x=true_value, color="yellow", linestyle="-")
        axs[i // 2, i % 2].text(true_value, 0.5, "True Value", rotation=90)

        axs[i // 2, i % 2].set_title(f"Decoder {decoder}")
        axs[i // 2, i % 2].legend()

    figure_name = f"{json_file_path_blue_top.stem.replace('.json', '')} and {json_file_path_red_top.stem.replace('.json', '')}"

    # add a title for the whole figure
    fig.suptitle(f"Results for {figure_name}"),

    # increase the space between the subplots
    fig.tight_layout()

    # save the figure
    fig.savefig(json_file_path_blue_top.parent / f"{figure_name}.png")


if __name__ == "__main__":
    import os

    # test_data_base_folder = Path("data", "low_level_vision", "jastrow_test")
    # test_data_folders = os.listdir(test_data_base_folder)

    # for folder in test_data_folders:
    #     decoder_evaluate(
    #         save_name=folder,
    #         results_folder="./results/jastrow/",
    #         dataset_folder=f"data/low_level_vision/jastrow_test/{folder}/",
    #         pretraining="models/jastrow/linear_decoder.pt",
    #         gpu_num=0,
    #         batch_size=32,
    #         use_residual_decoder=True,
    #     )

    json_results_folder_base = Path("results", "jastrow")
    json_results_files = os.listdir(json_results_folder_base)
    json_results_files = [i for i in json_results_files if i.endswith(".json")]

    for i in ["0", "0.05", "0.1", "0.15", "0.2", "0.25"]:
        analyse(
            json_results_folder_base / f"blue_on_top_{i}_smaller.json",
            json_results_folder_base / f"red_on_top_{i}_smaller.json",
        )
