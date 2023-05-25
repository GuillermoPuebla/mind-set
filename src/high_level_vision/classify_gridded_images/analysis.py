import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

folder = "results/distance_similarity/high_level_vision/classify_gridded_images/grid_degree45/gsize16/ImageNet/dataframe.pickle"
color_cycle = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)

d = pickle.load(open(folder, "rb"))
d["dataframe"]

df = d["dataframe"]
layers_names = d["layers_names"]
np.any(np.logical_and(df["base_obj"] == "Airplane", df["obj"] == "zebra"))

# for each object, check whether the similarity between del and del complement is higher than with same del for any other object.
# First the distance between same objects at different conditions
df_same_obj = (
    df[df["base_obj"] == df["obj"]].groupby("level")[layers_names[-1]].apply(list)
)


sns.kdeplot(
    df_same_obj["del"], color=color_cycle[0], label="same object, same deletion"
)
# sns.kdeplot(df_same_obj['del_2shift'], color=color_cycle[1], label='same object, shifted deletion')
sns.kdeplot(
    df_same_obj["del_complement"],
    color=color_cycle[2],
    label="same object, complement deletion",
)

plt.legend()
plt.show()

mm = df[df["base_obj"] == df["obj"]].groupby("level")[layers_names[-1]].mean()
std = df[df["base_obj"] == df["obj"]].groupby("level")[layers_names[-1]].std()
plt.bar(range(len(mm)), mm, yerr=std)
plt.show()

############
# Now the distance between different objects at different conditions
df_diff_obj = (
    df[df["base_obj"] != df["obj"]].groupby("level")[layers_names[-1]].apply(list)
)


sns.kdeplot(
    df_diff_obj["del"], color=color_cycle[0], label="diff object, same deletion"
)
# sns.kdeplot(df_diff_obj['del_2shift'], color=color_cycle[1], label='diff object, shifted deletion')
sns.kdeplot(
    df_diff_obj["del_complement"],
    color=color_cycle[2],
    label="dff object, complement deletion",
)

plt.legend()
plt.show()

mm = df[df["base_obj"] != df["obj"]].groupby("level")[layers_names[-1]].mean()
std = df[df["base_obj"] != df["obj"]].groupby("level")[layers_names[-1]].std()
plt.bar(range(len(mm)), mm, yerr=std)
plt.show()
