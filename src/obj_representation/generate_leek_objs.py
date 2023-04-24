"""
Generate directories for doing distance similarity analysis with the Leek Dataset in
The Structure of Three-Dimensional Object Representations in Human Vision: Evidence From Whole–Part Matching
by Leek, reppa, Arguin, 2005
You need to files "data/obj_representation/imgs_upscaled_mod.png" and "data/obj_representation/additional_mask.png".
To generate these files starting fomr the pdf we followed these steps:
    1. Convert the PDF page 3 to an image
    2. Upscale the PNG. We use this online service https://www.upscale.media/upload, AI upscaling 2X with enhance quality: ON
    3. Convert the upscaled image to SVG through this online service https://convertio.co/png-svg/
    4. Convert the SVG image into PNG https://convertio.co/png-svg/ select DPI=200. This should give a high resolution version of the original image.
    5. Now we use paint.net to fix few little things: few "spots" that seem to not belong to any objects. We move the "Samll" objects on the right by around 3500 pixels. This is the imgs_upscaled_mod.png
    6. Copy imgs_upscaled_mod.png -> additional_mask.png. In this file, every object that has parts that are too far away from one another needs to be covered with a black "mask". Almost all of them are from the "open contour" for small objects.
"""

import os
import pathlib
import cv2
import numpy as np
from skimage import measure

from tqdm import tqdm
import PIL.Image as Image


def get_objects_from_image(path_to_image, additional_mask_path):
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    additional_mask = cv2.imread(additional_mask_path, cv2.IMREAD_GRAYSCALE)

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
    np.unique(labels)
    cc = 0
    for label in tqdm(np.unique(labels)):
        print(cc)
        if label == 0:  # Skip the background
            continue

        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255

        object_np = cv2.bitwise_and(
            inverted_binary_image, inverted_binary_image, mask=mask
        )

        object_rows, object_cols = np.where(object_np != 0)

        # Find the bounding box of the object
        min_row, max_row = np.min(object_rows), np.max(object_rows)
        min_col, max_col = np.min(object_cols), np.max(object_cols)
        ct_y, ct_x = (
            min_row + (max_row - min_row) / 2,
            min_col + (max_col - min_col) / 2,
        )
        delimiter_main_obj = np.array(
            [816, 1666, 2385, 3250, 4042, 4907, 5581, 6241, 7169, 8039, 8865, 9540]
        )
        name = [
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
        delimiter_type = [1000, 1734, 2311, 2900, 3400, 3870, 4432, 5000, 5497]

        type = np.where(delimiter_type - ct_x > 0)[0][0]
        main_obj_type = np.where(delimiter_main_obj - ct_y > 0)[0][0]

        # Crop the image using the bounding box coordinates
        cropped_image = cv2.bitwise_not(image)[
            min_row : max_row + 1, min_col : max_col + 1
        ]

        path = f"data/obj_representation/{name[type]}/{main_obj_type}.png"
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        Image.fromarray(cropped_image).save(path)
        cc += 1


if __name__ == "__main__":
    image_path = "data/obj_representation/imgs_upscaled_mod.png"
    mask = "data/obj_representation/additional_mask.png"
    get_objects_from_image(image_path, mask)
