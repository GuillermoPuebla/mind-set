import os
import re
from copy import deepcopy

import PIL.Image as Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F


class PasteOnCanvas(torch.nn.Module):
    def __init__(self, canvas_to_image_ratio, background):
        super().__init__()
        self.canvas_to_image_ratio = canvas_to_image_ratio
        self.background = background

    def forward(self, pil_image):
        canvas = Image.new(
            "RGBA",
            (
                int(np.max(pil_image.size) * self.canvas_to_image_ratio),
                int(np.max(pil_image.size) * self.canvas_to_image_ratio),
            ),
            self.background,
        )
        canvas = paste_at_center(canvas, pil_image).convert("RGB")
        return canvas


def get_affine_rnd_fun_from_code(transf_values):
    tr = (
        (
            lambda: [
                np.random.uniform(*transf_values["translation"]),
                np.random.uniform(*transf_values["translation"]),
            ]
        )
        if "translation" in transf_values and transf_values["translation"]
        else lambda: (0, 0)
    )

    scale = (
        (lambda: np.random.uniform(*transf_values["scale"]))
        if "scale" in transf_values and transf_values["scale"]
        else lambda: 1.0
    )
    rot = (
        (lambda: np.random.uniform(*transf_values["rotation"]))
        if "rotation" in transf_values and transf_values["rotation"]
        else lambda: 0
    )
    return lambda: {"rt": rot(), "tr": tr(), "sc": scale(), "sh": 0.0}


def my_affine(img, translate, **kwargs):
    return F.affine(
        img,
        translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])],
        **kwargs,
    )


def save_figs(path, set, extra_info="", n=None):
    num_rows = len(set) if n is None else n
    fig, ax = plt.subplots(num_rows, 2)

    if np.ndim(ax) == 1:
        ax = np.array([ax])

    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])

    # Calculate DPI based on image dimensions to avoid downscaling
    img_width, img_height = set[0][0].shape[1], set[0][0].shape[0]
    total_width = img_width * 2
    total_height = img_height * num_rows

    dpi = img_width
    figsize = total_width / float(dpi), total_height / float(dpi)

    plt.gcf().set_size_inches(figsize)
    plt.gcf().set_dpi(dpi)

    plt.suptitle(f"Size: f{set[0][0].shape}\n{extra_info}")

    [x.set_xticks([]) for x in ax.flatten()]
    [x.set_yticks([]) for x in ax.flatten()]

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0.1)

    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(plt.gcf())


def has_subfolders(folder_path):
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):
            return True
    return False


def paste_at_center(canvas, image_to_paste):
    canvas = deepcopy(canvas)
    # Calculate the center of the canvas
    canvas_width, canvas_height = canvas.size
    canvas_center_x = canvas_width // 2
    canvas_center_y = canvas_height // 2

    # Calculate the position to paste the image so its center aligns with the canvas center
    image_width, image_height = image_to_paste.size
    position_x = canvas_center_x - (image_width // 2)
    position_y = canvas_center_y - (image_height // 2)
    position = (position_x, position_y)

    # Paste the image onto the canvas
    canvas.paste(image_to_paste, position)
    return canvas
