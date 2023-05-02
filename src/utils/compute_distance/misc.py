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

def get_new_affine_values(transf_code):
    # Example transf_code = 's[0.2, 0.3]tr[0,90]'  -> s is within 0.2, 0.3, t is default, r is between 0 and 90 degrees
    def get_values(code):
        real_num = r"[-+]?[0-9]*\.?[0-9]+"
        try:
            return [
                float(i)
                for i in re.search(
                    f"{code}\[({real_num}),\s?({real_num})]", transf_code
                ).groups()
            ]
        except AttributeError:
            if code == "t":
                return [-0.2, 0.2]
            if code == "s":
                return [0.7, 1.3]
            if code == "r":
                return [0, 360]

    tr = (
        [np.random.uniform(*get_values("t")), np.random.uniform(*get_values("t"))]
        if "t" in transf_code
        else (0, 0)
    )
    scale = np.random.uniform(*get_values("s")) if "s" in transf_code else 1.0
    rot = np.random.uniform(*get_values("r")) if "r" in transf_code else 0
    return {"rt": rot, "tr": tr, "sc": scale, "sh": 0.0}


def my_affine(img, translate, **kwargs):
    return F.affine(
        img,
        translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])],
        **kwargs,
    )


def save_figs(path, set, extra_info="", n=None):
    fig, ax = plt.subplots(len(set) if n is None else n, 2)
    if np.ndim(ax) == 1:
        ax = np.array([ax])
    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])
    # [x.axis('off') for x in ax.flatten()]
    plt.gcf().set_size_inches([2.4, 5])
    plt.suptitle(f"Size: f{set[0][0].shape}\n{extra_info}")

    [x.set_xticks([]) for x in ax.flatten()]
    [x.set_yticks([]) for x in ax.flatten()]

    plt.savefig(path)


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
