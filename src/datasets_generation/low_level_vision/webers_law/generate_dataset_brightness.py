from cgi import test
import math
import os
import errno
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import itertools
import random

from torch import rand

from PIL import Image, ImageDraw
import math


def draw_line(length, width_range, lum_range, len_var, unit, size_imx, size_imy):
    im = Image.new("RGB", (size_imx, size_imy), color="black")

    ### Randomly draw width and luminance from range
    width = np.random.randint(width_range[0], width_range[1])
    lum = np.random.randint(lum_range[0], lum_range[1])
    delta_len = np.random.randint(len_var[0], len_var[1])
    len_in_pix = length * unit
    new_len_in_pix = len_in_pix + delta_len

    ### Find coordinates of line in the middle of image
    xc = size_imx / 2
    yc = size_imy / 2
    x0, y0 = xc - (new_len_in_pix / 2), yc
    x1, y1 = xc + (new_len_in_pix / 2), yc
    bbox = [(x0, y0), (x1, y1)]

    drawing = ImageDraw.Draw(im)
    drawing.line(bbox, width=width, fill=(lum, lum, lum))

    return im


def draw_ellipse(brightness, bright_unit, offset, radius_range, size_imx, size_imy):
    ### Create image with black as background
    im = Image.new("L", (size_imx, size_imy), color=0)
    drawing = ImageDraw.Draw(im)

    ### Convert brightness (percent) into a gray value [0-256)
    im_color = (
        brightness * bright_unit
    ) + offset  # scales 'brightness' but still in percent
    gray_val = int((im_color / 100) * 255)

    radius = np.random.randint(
        radius_range[0], radius_range[1]
    )  # randomly sample a radius from the range

    ### Determine coordinates of stimulus "disk"
    xc = size_imx / 2
    yc = size_imy / 2
    x0, y0 = xc - radius, yc - radius
    x1, y1 = xc + radius, yc + radius
    bbox = [(x0, y0), (x1, y1)]

    ### Draw an stimulus disk of brightness gray_val
    drawing.ellipse(bbox, outline=None, fill=gray_val)

    return im


def gen_stim(stim_type="length", level=1, location=(0, 0), params=None):
    """
    Generate a stimulus to test Weber's law
        stim_type = 'length' / 'brightness' / 'numerosity'
        category = stimulus category (determines intensity)
        params = various parameters of image / shape generation
    """
    size_imx = params["size_x"]
    size_imy = params["size_y"]

    im = draw_line(
        length=level,
        width_range=params["line_width_range"],
        lum_range=params["line_lum_range"],
        len_var=params["length_var"],
        unit=params["length_unit"],
        pdash=params["pdash"],
        size_imx=size_imx,
        size_imy=size_imy,
    )
    img = translate_rotate(
        im, location=location, max_rot=params["max_rot"], flip90=params["flip90"]
    )  # transformed image
    # elif stim_type == "brightness":
    #     im = draw_ellipse(
    #         brightness=level,
    #         bright_unit=params["bright_unit"],
    #         offset=params["offset"],
    #         radius_range=params["radius_range"],
    #         size_imx=size_imx,
    #         size_imy=size_imy,
    #     )
    #     imt = translate_rotate(
    #         im, location=location, max_rot=params["max_rot"], fillcolor="black"
    #     )  # transformed image

    return img


def translate_rotate(im, location, max_rot, fillcolor="black", flip90=False):
    """Generate a random translation and rotation of image"""
    if flip90 == True:
        ### Randomly chooose between 0 and 90 degree flip
        flip = np.random.randint(0, 2)  # flip is either 0 or 1
        theta = flip * 90
    else:
        theta = np.random.randint(-max_rot, max_rot + 1)  # random rotation
    tx_ii = location[0]  # random translation
    ty_ii = location[1]

    ### First translate, then rotate - to ensure rotated segment remains completely on the canvas
    im = im.rotate(
        angle=0, translate=(tx_ii, ty_ii), resample=Image.BILINEAR, fillcolor=fillcolor
    )
    newim = im.rotate(angle=theta, resample=Image.BILINEAR, fillcolor=fillcolor)
    return newim


def get_params(stim_type):
    params = {}

    ### Image related parameters
    scale = 1  # 1 means 224 x 224
    params["scale"] = scale
    params["size_x"] = scale * 224
    params["size_y"] = scale * 224

    ### Stimuli related parameters
    params["stim_types"] = ["length"]  # ['length', 'brightness', 'numerosity']
    params["decoder_type"] = "regr"  # 'regr' / 'class'

    ### Intensity related parameters for testing Weber's law
    params["test_intens"] = [
        5,
        10,
        15,
    ]  # Mean intensities at which variation is compared
    params["categories"] = [(2 * ii) + 1 for ii in range(10)]

    # Parameters related to testing 'length'
    params["length_unit"] = 8  # pixels

    ### Parmeters related to testing 'brightness'
    params["bright_unit"] = 3  # luminosity expressed as percent (20% becomes 80%)
    params[
        "offset"
    ] = 20  # minimum brightness (some studies (e.g. Barlow, 1956) show Weber's Law doesn't hold for very dark values)

    params["radius_range"] = [
        int(params["size_x"] / 20),
        int(params["size_x"] / 6),
    ]  # [min, max) radius of patch

    ###  Following parameters are introduced to increase data variability along irrelevant (for length) dimensions
    params["line_width_range"] = [1, 5]  # range of line widths in pixels [low, high)
    params["line_lum_range"] = [
        100,
        256,
    ]  # range of brightness values in range 0--255 [low, high)
    params["length_var"] = [
        0,
        1,
    ]  # range of length variability [low, high) in pixels (Note: this should be less than 'length_unit')
    params[
        "max_rot"
    ] = 0  # image will be rotated by random rotation in the range [-max_rot, +max_rot] (degrees)

    if stim_type == "length":
        params["max_trans"] = [
            int(
                (params["size_x"] / 2)
                - (
                    (params["categories"][-1] / 2) * params["length_unit"]
                    + params["line_width_range"][-1]
                )
            ),
            int((params["size_y"] / 2) - params["line_width_range"][-1]),
        ]
    elif stim_type == "brightness":
        params["max_trans"] = [
            int(params["size_x"] / 2) - params["radius_range"][1],
            int(params["size_y"] / 2) - params["radius_range"][1],
        ]

    return params


# stim_types= length, numerosity, brightness
def create_set(params, stim_types, set_type="train"):
    # stim_types = params["stim_types"]
    # intensities = params["test_intens"]
    # (train_samples, test_samples) = get_train_test_split(
    #     params["max_trans"], ntrain=ntrain, ntest=ntest
    # )

    for stim_id, stim_name in enumerate(stim_types):
        print(
            "Creating {0} images for {1} stimuli             ".format(
                set_type, stim_name
            ),
            end="\r",
        )
        niter = 10
        # for (intens_id, intens_val) in enumerate(intensities):
        cats = params["categories"]
        for cc in cats:
            for ii in range(niter):
                im = gen_stim(
                    stim_type=stim_name, level=cc, location=(0, 0), params=params
                )


if __name__ == "__main__":
    create_set(
        get_params(stim_type="numerosity"),
        "train",
    )
    create_set(get_params(), "test")
    print("\n")
