from IPython.display import display, Markdown
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def open_df(dataset_name, pretraining):
    folder = Path("results") / dataset_name / pretraining
    df = pd.read_csv(folder / "dataframe.csv")
    with open(folder / "layers.txt", "r") as f:
        layers_names = f.readlines()
    layers_names = [layer.strip() for layer in layers_names]
    return df, layers_names


def run_all_layers_analysis(
    dataset_name, pretraining, list_comparison_levels=None, ylim=None, xlim=None
):
    df, _ = open_df(dataset_name, pretraining)

    if list_comparison_levels is not None:
        df = df[df["ComparisonLevel"].isin(list_comparison_levels)]

    mean_distances = df.groupby("ComparisonLevel").mean()

    plt.figure(figsize=(25, 10))
    for c in mean_distances.transpose().columns:
        plt.plot(mean_distances.transpose()[c], label=c)
    first_linear_idx = next(
        i for i, layer in enumerate(mean_distances.columns) if "Linear" in layer
    )
    plt.ylim(ylim) if ylim is not None else None

    plt.xticks(
        [0, first_linear_idx], ["0:Conv2D", mean_distances.columns[first_linear_idx]]
    )
    plt.xlim(xlim) if xlim is not None else None

    plt.title(f"Average Euclidean Distance for All Layers")
    plt.xlabel("Layers")
    plt.ylabel("Average Euclidean Distance")
    plt.legend()
    plt.show()


def run_standard_analysis_one_layer(
    dataset_name: Path, pretraining, idx_layer_used, list_comparison_levels=None
):
    df, layers_names = open_df(dataset_name, pretraining)
    if list_comparison_levels is not None:
        df = df[df["ComparisonLevel"].isin(list_comparison_levels)]
        # df.set_index("ComparisonLevel", inplace=True)
        # df = df.reindex(list_comparison_levels)
        # df.reset_index(inplace=True)

    num_layers = len(layers_names) - 1

    display(
        Markdown(
            f"# Analysis for dataset ***{dataset_name}***\n## Pretraining: ***{pretraining}***, LayerIdx: ***{idx_layer_used}***"
        )
    )

    name_layer_used = layers_names[idx_layer_used]

    mean_distances = df.groupby("ComparisonLevel").mean()
    mean_distances_std = df.groupby("ComparisonLevel").std()

    mean_distances_t_targeted_layer = mean_distances[name_layer_used].transpose()
    mean_distances_t_targeted_layer_std = mean_distances_std[
        name_layer_used
    ].transpose()

    if list_comparison_levels is not None:
        mean_distances_t_targeted_layer = mean_distances_t_targeted_layer.reindex(
            list_comparison_levels
        )
        mean_distances_t_targeted_layer_std = (
            mean_distances_t_targeted_layer_std.reindex(list_comparison_levels)
        )

    ##############
    plt.figure(figsize=(5, 6))
    plt.bar(
        mean_distances_t_targeted_layer.index,
        mean_distances_t_targeted_layer.values,
        yerr=mean_distances_t_targeted_layer_std,
        color=["blue", "orange"],
    )
    # plt.xticks(rotation=90)
    plt.title(
        f"Average Euclidean Distance for Layer {idx_layer_used}/{num_layers}: {name_layer_used}"
    )
    plt.xlabel("Comparison Level")
    plt.ylabel("Average Euclidean Distance")
    plt.show()

    ##
    from statsmodels.stats.anova import AnovaRM

    r = AnovaRM(
        data=df,
        depvar=name_layer_used,
        subject="MatchingLevels",
        within=["ComparisonLevel"],
        aggregate_func="mean",
    ).fit()

    print(r.anova_table)

    grouped = df.groupby(["ComparisonLevel", "MatchingLevels"]).mean()
    std_dev = df.groupby(["ComparisonLevel", "MatchingLevels"]).std()

    # Reset the index
    grouped = grouped.reset_index()
    std_dev = std_dev.reset_index()

    # Perform the operations conditionally
    if grouped["MatchingLevels"].str.strip("[]").str.isdigit().all():
        grouped["MatchingLevels"] = (
            grouped["MatchingLevels"].str.strip("[]").astype(int)
        )
        grouped = grouped.sort_values("MatchingLevels")

        std_dev["MatchingLevels"] = (
            std_dev["MatchingLevels"].str.strip("[]").astype(int)
        )
        std_dev = std_dev.sort_values("MatchingLevels")

    # Set the index again
    grouped = grouped.set_index(["ComparisonLevel", "MatchingLevels"])
    std_dev = std_dev.set_index(["ComparisonLevel", "MatchingLevels"])

    last_layer = grouped[name_layer_used].unstack()
    last_layer_std = std_dev[name_layer_used].unstack()

    last_layer_transposed = last_layer.transpose()

    ax = last_layer_transposed.plot(
        kind="bar", yerr=last_layer_std.transpose(), figsize=(20, 6), capsize=0
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.title(
        f"Average Euclidean Distance for Layer {idx_layer_used}/{num_layers}: {name_layer_used}"
    )

    plt.xlabel("Sample Name")
    plt.ylabel("Average Euclidean Distance")
    plt.legend(title="Comparison Level")
    plt.show()
