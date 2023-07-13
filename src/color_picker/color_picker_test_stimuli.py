"""
This script draws two boxes with different shades of grey.ƒ
"""

from PIL.Image import new
from PIL import Image, ImageDraw
from PIL.Image import Resampling
import numpy as np
from pathlib import Path


# # configs
# self.target_image_width = 224
# self.initial_expansion = 4  # alias, larger the more aliasing
# self.train_test_ratio = 0.8  # float between 0 and 1, larger the more training data
# self.progress_refresh_probability = 1000  # the larger the less frequent the progress bar is updated

# # configs for the arrow
# self.arrow_line_length = 45  # pixels
# self.triangle_height = 30  # pixels
# self.triangle_width = 20  # pixels

# on a image of size 224 * 4, the arrow line length is 45 pixels


def add_arrow(canvas, coord: tuple[float, float], fill=255):
    image_width = canvas.size[0]
    resize_ratio = image_width / (4 * 224)

    # configs for the arrow
    arrow_line_length = 45 * resize_ratio
    triangle_height = 30 * resize_ratio
    triangle_width = 20 * resize_ratio
    line_width = int(2 * resize_ratio)

    if 0 < coord[0] < 1 and 0 < coord[1] < 1:
        coord = tuple(map(lambda x: x * image_width, coord))

    coord = (coord[0], coord[1] - 1)

    coord_1 = coord
    coord_2 = (coord[0], coord[1] - arrow_line_length)

    coord_triangle_left = (coord[0] - triangle_width, coord[1] - triangle_height)
    coord_triangle_right = (coord[0] + triangle_width, coord[1] - triangle_height)

    draw = ImageDraw.Draw(canvas)
    draw.line((coord_1, coord_2), fill=fill, width=line_width)
    draw.polygon((coord_1, coord_triangle_left, coord_triangle_right), fill=fill)
    return canvas


if __name__ == "__main__":
    import os
    from PIL import Image, ImageDraw as Draw, ImageOps
    from pathlib import Path
    from tqdm import tqdm

    save_path = Path("data", "checkerboard")
    img = Image.open(Path("assets", "checkerboard.png"))

    os.makedirs(save_path, exist_ok=True)
    color_of_interest = (
        120,
        120,
        120,
    )  # get this from the image of interest using the color selector script under the same repository

    width, height = img.size
    coordinates = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if img.getpixel((x, y))[: len(color_of_interest)] == color_of_interest
    ]

    for coordinate in tqdm(coordinates, colour="green"):
        img_copy = img.copy().convert("L")
        img_copy = add_arrow(img_copy, tuple(coordinate))
        img_copy.thumbnail((224, 224), Image.Resampling.LANCZOS)
        img_copy.save(save_path / f"{coordinates.index(coordinate)}.png")
