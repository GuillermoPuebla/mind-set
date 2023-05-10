import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="white")

folder = "results/distance_similarity/high_level_vision/"
folder_leek = folder + "leek_reppa_arguin_2005_dataset"

pk = pickle.load(open(folder_leek + "/ImageNet/dataframe.pickle", "rb"))

color_cycle = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)

df, layers_names = pk["dataframe"], pk["layers_names"]
df = df[df["level"] != "base"]
ll = df[["level"] + layers_names]


def split_levels(level):
    size, type_ = level.split("_")
    return pd.Series({"size": size, "type": type_})


new_df = pd.concat([df, df["level"].apply(split_levels)], axis=1).drop("level", axis=1)


## All data  collapsed across object, last layer
grouped_by_type = new_df.groupby(["type"])[layers_names[-1]].apply(list)
[sns.kdeplot(grouped_by_type[i], label=i) for i in grouped_by_type.keys()]
plt.legend()
plt.xlabel("Euclidean Distance")
plt.ylabel("f(x)")
plt.show()

##
grouped_type_set = new_df.groupby(["type", "set"])[layers_names[-1]].apply(list)
[
    sns.kdeplot(grouped_type_set["vol"][i], color=color_cycle[0])
    for i in grouped_type_set["vol"].keys()
]

[
    sns.kdeplot(grouped_type_set["closed"][i], color=color_cycle[1])
    for i in grouped_type_set["closed"].keys()
]

[
    sns.kdeplot(grouped_type_set["surf"][i], color=color_cycle[2])
    for i in grouped_type_set["surf"].keys()
]
plt.show()

##
grouped_set_type = new_df.groupby(["set", "type"])[layers_names[-1]].mean()
pivot_df = grouped_set_type.reset_index().pivot(index="set", columns="type")
pivot_df.plot(kind="bar")
plt.show()

##
from statsmodels.stats.anova import AnovaRM

r = AnovaRM(
    data=new_df,
    depvar=layers_names[-1],
    subject="set",
    within=["size", "type"],
    aggregate_func="mean",
).fit()

r.anova_table


## Post hoc analysis
from statsmodels.stats.multitest import multipletests

unique_types = new_df["type"].unique()
pairwise_comparisons = pd.DataFrame()


for type1 in unique_types:
    for type2 in unique_types:
        if type1 < type2:
            temp_df = new_df[(new_df["type"] == type1) | (new_df["type"] == type2)]
            r_pairwise = AnovaRM(
                data=temp_df,
                depvar=layers_names[-1],
                subject="set",
                within=["size", "type"],
                aggregate_func="mean",
            ).fit()
            p_value = r_pairwise.anova_table["Pr > F"][0]
            pairwise_comparisons = pairwise_comparisons.append(
                pd.DataFrame({"type1": [type1], "type2": [type2], "p_value": [p_value]})
            )

# Bonferroni correction
reject, pvals_corrected, _, _ = multipletests(
    pairwise_comparisons["p_value"], alpha=0.01, method="bonferroni"
)
pairwise_comparisons["p_value_bonferroni"] = pvals_corrected
pairwise_comparisons["reject"] = reject

print(pairwise_comparisons)
mean_performances = new_df.groupby("type")[layers_names[-1]].mean()

# To display the mean performances for each type
print(mean_performances)


mean_performances = new_df.groupby("type")[layers_names[-1]].mean()
sem_performances = new_df.groupby("type")[layers_names[-1]].sem()

# Create a DataFrame with mean and standard error for each type
results = pd.DataFrame(
    {"Mean": mean_performances, "SEM": sem_performances}
).reset_index()

## Create a bar plot with error bars (notice that we use SEM not std)
plt.figure(figsize=(10, 6))
sns.barplot(x="type", y="Mean", data=results, yerr=results["SEM"], capsize=1)
plt.xlabel("Type")
plt.ylabel(f"Mean {layers_names[-1]}")
plt.title(f"Last layer with Standard Error")
plt.show()

##
