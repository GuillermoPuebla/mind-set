import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def normalized_to_grey(value: float):
    """Map (-1, 1) values to (0, 255) values."""
    return int((value + 1) * 127.5)


with open(Path("results", "results_checkerboard_residual.json"), "r") as f:
    results = json.load(f)

with open(Path("assets", "same_color_coordinates.json"), "r") as f:
    coordinates = json.load(f)  # used as an index

for i in results:
    i["coordinates"] = tuple(coordinates[int(i["label"])])
    # drop the label
    del i["label"]

# # read image
# img = Image.open(Path("assets") / "checkerboard_no_letters.png")
# img = img.convert("L")

# print(results[0])


# def get_decoder_prediction_image(decoder: int):
#     img_copy = img.copy()
#     results_copy = [i for i in results if i["decoder"] == decoder]

#     for i in results_copy:
#         img_copy.putpixel(i["coordinates"], normalized_to_grey(i["prediction"]))

#     img_copy.save(Path("results", f"model_prediction_decoder_{decoder}.png"))


# for i in range(6):
#     get_decoder_prediction_image(i)

# read the masks for the shapes


def plot_distribution_of_predictions(decoder: int):
    mask_1 = Image.open(Path("results", "mask_1.png"))
    mask_2 = Image.open(Path("results", "mask_2.png"))
    img = Image.open(Path("assets") / "checkerboard_no_letters.png")
    img = img.convert("L")
    img_copy = img.copy()
    dummy_img = Image.new("L", img.size, 0)
    results_copy = [i for i in results if i["decoder"] == decoder]

    for i in results_copy:
        dummy_img.putpixel(i["coordinates"], normalized_to_grey(i["prediction"]))
        img_copy.putpixel(i["coordinates"], normalized_to_grey(i["prediction"]))

    dummy_img.save(Path("results", f"model_prediction_decoder_{decoder}.png"))
    img_copy.save(Path("results", f"model_prediction_decoder_{decoder}_with_shapes.png"))

    # apply the masks
    dummy_img = np.array(dummy_img)
    mask_1 = np.array(mask_1)
    mask_2 = np.array(mask_2)

    # get the values for the two shapes
    shape_1_values = dummy_img[mask_1 == 255]
    shape_2_values = dummy_img[mask_2 == 255]

    # overlay the two shapes
    both = np.zeros_like(dummy_img)
    both[mask_1 == 255] = 255
    both[mask_2 == 255] = 255
    both_values = np.zeros_like(dummy_img)
    both_values[mask_1 == 255] = shape_1_values
    both_values[mask_2 == 255] = shape_2_values
    both_values = Image.fromarray(both_values)
    # crop to bbox
    bbox = Image.fromarray(both).getbbox()
    both_values = both_values.crop(bbox)
    both_values.save(Path("results", f"both_masks_decoder_{decoder}.png"))

    # t-test
    from scipy import stats

    t, p = stats.ttest_ind(shape_1_values, shape_2_values)

    # plot the distribution using curve plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    # plot the two distribution over each other
    sns.kdeplot(shape_1_values, shade=True, label="Shape 1 (on top)")
    sns.kdeplot(shape_2_values, shade=True, label="Shape 2 (in middle)")
    plt.title(f"Darker <-         Decoder {decoder}         -> Brighter")
    plt.legend()
    # add the t-test result
    plt.text(
        0.1,
        0.98,
        f"Decoder {decoder}: t = {t.__round__(3)}, p = {p.__round__(3)}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(Path("results", f"decoder_{decoder}_distribution.png"))
    plt.clf()


for i in range(6):
    plot_distribution_of_predictions(i)

# shape_1_values = []
# shape_2_values = []

# p_bar = tqdm(results, desc="Processing results")

# for i in p_bar:
#     if i["coordinates"] in indices_1:
#         shape_1_values.append(i["prediction"])
#     if i["coordinates"] in indices_2:
#         shape_2_values.append(i["prediction"])

#     p_bar.update(1)
#     p_bar.set_description(f"Processing results: {len(shape_1_values)} / {len(shape_2_values)}")

# print(shape_1_values)

# with open(Path("results", "shape_1_values.json"), "w") as f:
#     json.dump(shape_1_values, f)

# with open(Path("results", "shape_2_values.json"), "w") as f:
#     json.dump(shape_2_values, f)

# # read the values
# with open(Path("results", "shape_1_values.json"), "r") as f:
#     shape_1_values = json.load(f)

# with open(Path("results", "shape_2_values.json"), "r") as f:
#     shape_2_values = json.load(f)

# # test whether the two distributions are different significantly using a

# # plot the distribution using curve plot
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme(style="darkgrid")

# # plot the two distribution over each other
# sns.kdeplot(shape_1_values, shade=True, label="Shape 1")
# sns.kdeplot(shape_2_values, shade=True, label="Shape 2")

# plt.xlabel("Prediction")
# plt.ylabel("Density")

# # set legend
# plt.legend()

# plt.savefig(Path("distribution.png"))
