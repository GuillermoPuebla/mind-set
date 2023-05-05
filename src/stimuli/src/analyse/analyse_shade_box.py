from itertools import product
import numpy as np
from pathlib import Path
import json

# import seaborn as sns
import matplotlib.pyplot as plt

range_1 = np.arange(0, 255, 5)
range_2 = np.arange(0, 255, 5)
range_1 = [int(i) for i in range_1]
range_2 = [int(i) for i in range_2]
# combs = list(product(list(range(255)), list(range(255))))
combs = list(product(range_1, range_2))

# read data from results/boxes/results.json
path_to_results = Path("results") / "boxes_arrow" / "results.json"

with open(path_to_results, "r") as f:
    results = json.load(f)

for i in results:
    i["label"] = combs[int(i["label"])]
    i["inner"] = i["label"][0]
    i["outer"] = i["label"][1]

all_inner = list(set([i["inner"] for i in results]))
all_inner.sort()

# for value in all_inner:
#     baseline = [i for i in results if i["inner"] == value and i["outer"] == 0][0]["prediction"]
#     for j in results:
#         if j["inner"] == value:
#             j["prediction"] -= baseline


def plot_decoder_heatmap(decoder_id: int):
    relevant_results = [i for i in results if i["decoder"] == decoder_id]

    # plot the inner as x and outer as y and prediction as value in a heatmap
    inner = [i["inner"] for i in relevant_results]
    outer = [i["outer"] for i in relevant_results]
    prediction = [i["prediction"] for i in relevant_results]

    # create a 2d array of the predictions
    prediction_2d = np.zeros((len(range_1), len(range_2)))
    for i, j, k in zip(inner, outer, prediction):
        prediction_2d[range_1.index(i), range_2.index(j)] = k

    # plot the heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(prediction_2d)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(range_2)))
    ax.set_yticks(np.arange(len(range_1)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(range_2)
    ax.set_yticklabels(range_1)

    # legend
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Prediction", rotation=-90, va="bottom")

    # annotate axis
    ax.set_xlabel("Outer")
    ax.set_ylabel("Inner")

    ax.set_title(f"Decoder {decoder_id}")
    fig.tight_layout()
    # save the figure
    fig.savefig(f"plot/arrow_decoder_{decoder_id}_heatmap.png", dpi=300)


def plot_decoder_line(decoder_id: int):
    relevant_results = [i for i in results if i["decoder"] == decoder_id]

    # plot the inner as x and outer as y and prediction as value in a heatmap
    inner = [i["inner"] for i in relevant_results]
    outer = [i["outer"] for i in relevant_results]
    prediction = [i["prediction"] for i in relevant_results]

    # create a 2d array of the predictions
    prediction_2d = np.zeros((len(range_1), len(range_2)))
    for i, j, k in zip(inner, outer, prediction):
        prediction_2d[range_1.index(i), range_2.index(j)] = k

    # plot the line
    fig, ax = plt.subplots()
    for i in range(len(range_1)):
        ax.plot(range_2, prediction_2d[i, :], label=range_1[i])

    # legend
    ax.legend()

    # annotate axis
    ax.set_xlabel("Outer")
    ax.set_ylabel("Prediction")

    ax.set_title(f"Decoder {decoder_id}")
    fig.tight_layout()

    # save the figure
    fig.savefig(f"plot/line_arrow_decoder_{decoder_id}.png", dpi=300)


if __name__ == "__main__":
    for i in range(6):
        plot_decoder_line(i)
