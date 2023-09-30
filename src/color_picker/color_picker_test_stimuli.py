"""
This script draws two boxes with different shades of grey.ƒ
"""

from PIL import Image
from pathlib import Path
from src.color_picker.utils import add_arrow

import os
from tqdm import tqdm


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
