import os
import pathlib
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import glob
from PIL.ImageOps import grayscale, invert

import numpy as np

from src.utils.compute_distance.misc import paste_at_center


def apply_grid_mask(
    image_path,
    grid_size,
    grid_thickness=1,
    grid_shift=0,
    rotation_degrees=0,
    canvas_color=255,
):
    # Open the image
    image = grayscale(Image.open(image_path))
    scaleby = 100 / (np.max(image.size))
    image_resized = image.resize(
        (int(image.size[0] * scaleby), int(image.size[1] * scaleby))
    )
    # Convert the image to a NumPy array
    image_array = np.array(image_resized)
    # line_drawing_pixels = np.all(image_array == line_color, axis=-1)

    # Get image dimensions
    height, width = image_array.shape

    # Create a mask of zeros with the same dimensions as the image
    mask = np.full((height * 2, width * 2), False)

    # Set the grid mask values to 1
    for i in range(grid_shift, mask.shape[0], grid_size):
        mask[i : i + grid_thickness, :] = 1

    # for j in range(grid_shift, width, grid_size):
    #     mask[:, j : j + grid_thickness] = 1
    rotated_mask = np.array(
        Image.fromarray(mask).rotate(rotation_degrees, expand=True, fillcolor=(0))
    )
    # Image.fromarray(rotated_mask).show()
    rotated_mask = rotated_mask[
        rotated_mask.shape[0] // 2
        - height // 2 : rotated_mask.shape[0] // 2
        - height // 2
        + height,
        rotated_mask.shape[1] // 2
        - width // 2 : rotated_mask.shape[1] // 2
        - width // 2
        + width,
    ]
    # Image.fromarray(rotated_mask2).show()

    # intersection_pixels = np.logical_and(line_drawing_pixels, mask)
    masked_image = deepcopy(image_array)
    masked_image[rotated_mask] = canvas_color
    complement_image = deepcopy(image_array)
    complement_image[~rotated_mask] = canvas_color
    # Apply the mask to the image
    # masked_image_array = np.logical_and(image_array, mask)

    # Convert the masked NumPy array back to an Image object
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    complement_image = Image.fromarray(complement_image.astype(np.uint8))

    return masked_image, complement_image, mask


path = "data/obj_representation/baker_2018/outline_images_fix"
all_files = glob.glob(path + "/**")


# Example usage:
def create_dataset(grid_degree, grid_size, grid_thickness):
    result_folder = (
        f"data/obj_representation/baker_2018/grid_degree{grid_degree}/gsize{grid_size}/"
    )
    [
        pathlib.Path(result_folder + f"{ff}").mkdir(parents=True, exist_ok=True)
        for ff in ["del", "del_complement", f"del_{grid_size // 4}shift/"]
    ]
    background = Image.new("RGBA", (224, 224), (255, 255, 255))

    for f in all_files:
        masked_image, complement_image, mask = apply_grid_mask(
            f,
            grid_size,
            grid_shift=0,
            grid_thickness=grid_thickness,
            rotation_degrees=grid_degree,
            canvas_color=255,
        )

        masked_image = invert(paste_at_center(background, masked_image).convert("RGB"))
        masked_image.save(
            result_folder + "/del/" + os.path.splitext(os.path.basename(f))[0] + ".png"
        )

        complement_image = invert(
            paste_at_center(background, complement_image).convert("RGB")
        )
        complement_image.save(
            result_folder
            + "/del_complement/"
            + os.path.splitext(os.path.basename(f))[0]
            + ".png"
        )

        for f in all_files:
            masked_image, complement_image, mask = apply_grid_mask(
                f,
                grid_size,
                grid_shift=grid_size // 4,
                grid_thickness=grid_thickness,
                rotation_degrees=grid_degree,
                canvas_color=255,
            )
            masked_image = invert(
                paste_at_center(background, masked_image).convert("RGB")
            )

            masked_image.save(
                result_folder
                + f"/del_{grid_size // 4}shift/"
                + os.path.splitext(os.path.basename(f))[0]
                + ".png"
            )


image_size = 100
create_dataset(grid_size=8, grid_thickness=4, grid_degree=45)
create_dataset(grid_size=16, grid_thickness=9, grid_degree=45)