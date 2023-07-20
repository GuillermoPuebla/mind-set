import pathlib
import shutil
from typing import List

import PIL
import numpy as np
import sty
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter
import tqdm

try:
    import neptune
except:
    pass


def conditional_tqdm(iterable, enable_tqdm, **kwargs):
    if enable_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    else:
        return iterable


class ConfigSimple:
    def __init__(self, **kwargs):
        self.use_cuda = torch.cuda.is_available()
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]

    def __setattr__(self, *args, **kwargs):
        super().__setattr__(*args, **kwargs)


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)

    canvas_size = np.shape(image)

    font_scale = np.ceil(canvas_size[1]) / 150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[0, 0, 0],
        lineType=cv2.LINE_AA,
        thickness=6,
    )
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[255, 255, 255],
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    image = cv2.UMat.get(umat)
    image = np.array(image, np.uint8)
    return image


def weblog_dataset_info(
    dataloader,
    log_text="",
    dataset_name=None,
    weblogger=1,
    plotter=None,
    num_batches_to_log=2,
):
    stats = {}

    def simple_plotter(idx, data):
        images, labels, *more = data
        plot_images = images[0 : np.max((4, len(images)))]
        metric_str = "Debug/{} example images".format(log_text)
        lab = [f"{i.item():.3f}" for i in labels]
        if isinstance(weblogger, neptune.Run):
            [
                weblogger[metric_str].log(
                    File.as_image(
                        convert_normalized_tensor_to_plottable_array(
                            im, stats["mean"], stats["std"], text=lb
                        )
                        / 255
                    )
                )
                for im, lb in zip(plot_images, lab)
            ]

    if plotter is None:
        plotter = simple_plotter
    if "stats" in dir(dataloader.dataset):
        dataset = dataloader.dataset
        dataset_name = dataset.name_ds
        stats = dataloader.dataset.stats
    else:
        dataset_name = "no_name" if dataset_name is None else dataset_name
        stats["mean"] = [0.5, 0.5, 0.5]
        stats["std"] = [0.2, 0.2, 0.2]
        Warning(
            "MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch"
        )

    if isinstance(weblogger, neptune.Run):
        weblogger["Logs"] = f'{dataset_name} mean: {stats["mean"]}, std: {stats["std"]}'

    for idx, data in enumerate(dataloader):
        plotter(idx, data)
        if idx + 1 >= num_batches_to_log:
            break

    # weblogger[weblogger_text].log(File.as_image(image))


def imshow_batch(inp, stats=None, labels=None, title_more="", maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats["mean"]
        std = stats["std"]
    """Imshow for Tensor."""

    cols = int(np.ceil(np.sqrt(len(inp))))
    if ax is None:
        fig, ax = plt.subplots(cols, cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis("off")
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            text = (
                str(labels[idx]) + " " + (title_more[idx] if title_more != "" else "")
            )
            # ax[idx].set_title(text, size=5)
            ax[idx].text(
                0.5,
                0.1,
                f"{labels[idx]:.3f}",
                horizontalalignment="center",
                transform=ax[idx].transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0.01, left=0, right=1, hspace=0.2, wspace=0.01)
    return ax


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_lists_to_strings(obj):
    if isinstance(obj, list):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_strings(v) for k, v in obj.items()}
    else:
        return obj


DEFAULTS = {
    "canvas_size": (224, 224),
    "background_color": (0, 0, 0),
    "antialiasing": True,
    "regenerate": True,
}


def add_general_args(parser):
    parser.add_argument(
        "--output_folder",
        "-o",
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure used to create the dataset",
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default=DEFAULTS["canvas_size"],
        help="A string in the format NxM specifying the size of the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")])
        if isinstance(x, str)
        else x,
    )

    parser.add_argument(
        "--background_color",
        "-bg",
        default=DEFAULTS["background_color"],
        help="Specify the background as rgb value in the form R_G_B, or write [random] for a randomly pixellated background.or [rnd-uniform] for a random (but uniform) color",
        type=lambda x: (tuple([int(i) for i in x.split("_")]) if "_" in x else x)
        if isinstance(x, str)
        else x,
    )

    parser.add_argument(
        "--no_antialiasing",
        "-nantial",
        dest="antialiasing",
        help="Specify whether we want to disable antialiasing",
        action="store_false",
        default=DEFAULTS["antialiasing"],
    )

    parser.add_argument(
        "--no_regenerate_if_present",
        "-noreg",
        help="If the dataset is already present, DO NOT regenerate it, just return",
        dest="regenerate",
        action="store_false",
        default=DEFAULTS["regenerate"],
    )


def pretty_print_dict(dictionary, indent=0):
    key_color = sty.fg.blue
    value_color = sty.fg.green
    reset_color = sty.rs.fg

    for key, value in sorted(dictionary.items()):
        print(" " * indent + key_color + key + reset_color, end=": ")
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 4)
        else:
            print(value_color + str(value) + reset_color)


def update_dict(dictA, dictB, replace=True):
    for key in dictB:
        if (
            key in dictA
            and isinstance(dictA[key], dict)
            and isinstance(dictB[key], dict)
        ):
            # if the value in dictA is a dict and the value in dictB is a dict, recursively update the nested dict
            update_dict(dictA[key], dictB[key], replace)
        else:
            # otherwise, simply update the value in dictA with the value in dictB
            if replace or (not replace and key not in dictA):
                old_value = dictA[key] if key in dictA else "none"
                dictA[key] = dictB[key]
                print(
                    f"Updated {key} : {old_value} => {key}: {dictB[key]}"
                ) if old_value != dictB[key] else None
            else:
                print(
                    f"Value {key} not replaced as already present ({dictA[key]}) and 'replace=False'"
                )
    return dictA


def apply_antialiasing(img: PIL.Image, amount=None):
    if amount is None:
        amount = min(img.size) * 0.00334
    return img.filter(ImageFilter.GaussianBlur(radius=amount))


def delete_and_recreate_path(path: pathlib.Path):
    shutil.rmtree(path) if path.exists() else None
    path.mkdir(parents=True, exist_ok=True)
    print(sty.fg.yellow + f"Creating Dataset in {path}. Please wait..." + sty.rs.fg)
