import glob
import os
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt


def dotted_image(image_path, dot_distance=10, dot_size=1, max_side_length=224):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = resize_image(img, max_side_length)

    _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    contours, b = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dotted_img = np.zeros_like(img)

    def draw_dot(image, x, y, size, color=255):
        half_size = size // 2
        cv2.rectangle(
            image,
            (x - half_size, y - half_size),
            (x + half_size, y + half_size),
            color,
            -1,
        )

    for contour in contours:
        for i, point in enumerate(contour):
            # print(point)
            if i % dot_distance == 0:
                x, y = point[0]
                draw_dot(dotted_img, x, y, dot_size)

    return dotted_img


def resize_image(opencv_img, max_side_length=224):
    # Calculate new dimensions while keeping the aspect ratio
    height, width = opencv_img.shape
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = max_side_length
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_side_length
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(
        opencv_img, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_img


def center_image_on_canvas(opencv_img, canvas_size=224):
    # Load the image

    # Create a blank canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Calculate the center position on the canvas
    height, width = opencv_img.shape
    y_offset = (canvas_size - height) // 2
    x_offset = (canvas_size - width) // 2

    # Paste the image onto the canvas
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = opencv_img
    return canvas


from pathlib import Path


def create_dataset(dot_distance, dot_size):
    folder = Path("assets/baker_2018/outline_images_fix/")
    output_folder = (
        f"data/gestalt/good_continuation/dot_dist{dot_distance}_size{dot_size}/"
    )
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    all_images = glob.glob(str(folder) + "/**")
    for image in all_images:
        dotted_img = dotted_image(image, dot_distance=dot_distance, dot_size=dot_size)

        cv2.imwrite(
            output_folder + "/" + os.path.basename(image),
            center_image_on_canvas(dotted_img),
        )


if __name__ == "__main__":
    # not sure whether to use decoder or similarity approach here
    create_dataset(dot_distance=5, dot_size=2)
    create_dataset(dot_distance=10, dot_size=2)
    create_dataset(dot_distance=20, dot_size=2)
    create_dataset(dot_distance=30, dot_size=2)
    create_dataset(dot_distance=40, dot_size=2)
