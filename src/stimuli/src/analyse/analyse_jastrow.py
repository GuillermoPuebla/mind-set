import json
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


def analyse(result_folder="25_percent_smaller"):
    with open(Path("results", f"blue_{result_folder}", "results.json"), "r") as f:
        data_blue = json.load(f)

    with open(Path("results", f"red_{result_folder}", "results.json"), "r") as f:
        data_red = json.load(f)

    decoders = [i["decoder"] for i in data_blue]
    decoders = list(set(decoders))

    rmses = [0.076, 0.037, 0.029, 0.035, 0.1, 0.17]
    rmses = iter(rmses)
    true_value = 0.0

    # plot the results on overlapping distributions curves
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, decoder in enumerate(decoders):
        data_blue_decoder = [i["prediction"] for i in data_blue if i["decoder"] == decoder]
        data_red_decoder = [i["prediction"] for i in data_red if i["decoder"] == decoder]
        sns.kdeplot(data_blue_decoder, ax=axs[i // 2, i % 2], label="blue", color="blue")
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

    # add a title for the whole figure
    fig.suptitle(f"Results for {result_folder}")

    # increase the space between the subplots
    fig.tight_layout()

    # save the figure
    fig.savefig(f"{result_folder}.png")


if __name__ == "__main__":
    for i in ["0", "5", "10", "15", "20", "25"]:
        analyse(f"{i}_percent_smaller")
