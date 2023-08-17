"""
This script draws two boxes with different shades of grey.ƒ
"""

from PIL import Image, ImageDraw
from pathlib import Path

import os
from tqdm import tqdm


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


def generate_arrow_data(
    color_of_interest=(120, 120, 120),
    img=Path("assets", "checkerboard.png"),
    save_path: Path = Path("data", "checkerboard"),
    output_size=(224, 224),
):
    img = Image.open(img)
    os.makedirs(save_path, exist_ok=True)

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
        img_copy.thumbnail(output_size, Image.Resampling.LANCZOS)
        img_copy.save(save_path / f"{coordinates.index(coordinate)}.png")


if __name__ == "__main__":
    pass
