"""
Generate directories for doing distance similarity analysis with the Leek Dataset in
The Structure of Three-Dimensional Object Representations in Human Vision: Evidence From Whole–Part Matching
by Leek, reppa, Arguin, 2005
You need the files in "assets/leek_2005".
To generate these files starting fomr the pdf we followed these steps:
    1. Convert the PDF page 3 to an image
    2. Upscale the PNG. We use this online service https://www.upscale.media/upload, AI upscaling 2X with enhance quality: ON
    3. Convert the upscaled image to SVG through this online service https://convertio.co/png-svg/
    4. Convert the SVG image into PNG https://convertio.co/png-svg/ select DPI=200. This should give a high resolution version of the original image.
    5. Now we use paint.net to fix few little things: few "spots" that seem to not belong to any objects. We move the "Samll" objects on the right by around 3500 pixels. This is the imgs_upscaled_mod.png
    6. Copy imgs_upscaled_mod.png in additional_mask.png. In this file, every object that has parts that are too far away from one another needs to be covered with a black "mask". Almost all of them are from the "open contour" for small objects.
    Now you can run this script.
"""
import argparse
import csv
import os
import pathlib
from collections import deque

import cv2
import numpy as np
from skimage import measure

from tqdm import tqdm
import PIL.Image as Image

from src.utils.compute_distance.misc import paste_at_center
from src.utils.drawing_utils import (
    DrawShape,
    resize_image_keep_aspect_ratio,
    paste_linedrawing_onto_canvas,
)
from src.utils.misc import (
    apply_antialiasing,
    add_general_args,
    delete_and_recreate_path,
)


def generate_all(
    output_folder,
    canvas_size,
    background,
    antialiasing,
    object_longest_side,
):
    image_path = "assets/leek_2005//imgs_upscaled_mod.png"
    mask = "assets/leek_2005/additional_mask.png"
    categories = [
        "base",
        "large_open",
        "large_closed",
        "large_vol",
        "large_surf",
        "small_open",
        "small_closed",
        "small_vol",
        "small_surf",
    ]
    output_folder = (
        pathlib.Path("data") / "high_level_vision" / "volumetric_vs_surface"
        if output_folder is None
        else output_folder
    )
    delete_and_recreate_path(output_folder)
    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in categories]

    ds = DrawShape(
        background=background, canvas_size=canvas_size, antialiasing=antialiasing
    )
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    additional_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    _, binary_mask = cv2.threshold(additional_mask, 128, 255, cv2.THRESH_BINARY)

    inverted_binary_image = cv2.bitwise_not(binary_image)
    inverted_binary_mask = cv2.bitwise_not(binary_mask)

    kernel = np.ones((80, 80), np.uint8)
    opened_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)

    opened_img_and_mask = cv2.bitwise_or(opened_image, inverted_binary_mask)
    labels = measure.label(
        opened_img_and_mask,
        connectivity=1,
        background=0,
    )
    cc = 0
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "Background", "ObjectNum"])
        for label in tqdm(np.unique(labels)):
            # print(cc)
            if label == 0:  # Skip the background
                continue

            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == label] = 255

            object_np = cv2.bitwise_and(
                inverted_binary_image, inverted_binary_image, mask=mask
            )

            object_rows, object_cols = np.where(object_np != 0)

            min_row, max_row = np.min(object_rows), np.max(object_rows)
            min_col, max_col = np.min(object_cols), np.max(object_cols)
            ct_y, ct_x = (
                min_row + (max_row - min_row) / 2,
                min_col + (max_col - min_col) / 2,
            )
            delimiter_main_obj = np.array(
                [816, 1666, 2385, 3250, 4042, 4907, 5581, 6241, 7169, 8039, 8865, 9540]
            )

            delimiter_type = [1000, 1734, 2311, 2900, 3400, 3870, 4432, 5000, 5497]

            type = np.where(delimiter_type - ct_x > 0)[0][0]
            main_obj_type = np.where(delimiter_main_obj - ct_y > 0)[0][0]

            # Crop the image using the bounding box coordinates
            cropped_image = image[min_row : max_row + 1, min_col : max_col + 1]
            img = resize_image_keep_aspect_ratio(cropped_image, object_longest_side)
            pil_img = paste_linedrawing_onto_canvas(img, ds.create_canvas(), ds.fill)
            path = pathlib.Path(categories[type]) / f"{main_obj_type}.png"
            output_path = pathlib.Path(output_folder) / path

            (apply_antialiasing(pil_img) if antialiasing else pil_img).save(output_path)
            writer.writerow([path, categories[type], background, main_obj_type])

            cc += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=100,
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
