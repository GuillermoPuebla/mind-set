import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="white")

folder = "results/distance_similarity/high_level_vision/"
folder_leek = folder + "feature_deletion"

pk = pickle.load(open(folder_leek + "/ImageNet/dataframe.pickle", "rb"))

color_cycle = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)

df, layers_names = pk["dataframe"], pk["layers_names"]

df.groupby(["set", "level"])[layers_names[-1]].apply(list)

dfm = df.groupby(["set", "level"])[layers_names[-1]].mean()
dfs = df.groupby(["set", "level"])[layers_names[-1]].std()

dfm2 = dfm.reset_index()
dfm2["sme"] = dfs.reset_index()[layers_names[-1]]

pivot_df = dfm2.pivot_table(index="set", columns="level", values=["363: Linear", "sme"])

# Extract the 'yerr' values
yerr = pivot_df["sme"].values.T

## Create a barplot
plt.figure(figsize=(10, 6))
sns.barplot(x="set", y="363: Linear", hue="level", data=dfm2, yerr=yerr, capsize=0.1)
# plt.title("Barplot of 363: Linear by Set and Level with Error Bars")
plt.xlabel("Object")
plt.ylabel("Euclidean Distance")
plt.show()
##
