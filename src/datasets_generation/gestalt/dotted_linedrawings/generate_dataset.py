import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import sty

from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
    DEFAULTS,
)


class DrawDottedImage(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def dotted_image(self, image_path, dot_distance, dot_size):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, b = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dotted_img = np.ones_like(img) * 255

        def draw_dot(image, x, y, size, color):
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
                if i % dot_distance == 0:
                    x, y = point[0]
                    draw_dot(dotted_img, x, y, dot_size, color=0)

        mask = get_mask_from_linedrawing(dotted_img)

        stroke_canvas = self.create_canvas(background=self.fill, size=mask.size)
        canvas = self.create_canvas()
        canvas.paste(
            stroke_canvas,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas

    def center_image_on_canvas(self, opencv_img, canvas_size=224):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # Calculate the center position on the canvas
        height, width = opencv_img.shape
        y_offset = (canvas_size - height) // 2
        x_offset = (canvas_size - width) // 2

        # Paste the image onto the canvas
        canvas[y_offset : y_offset + height, x_offset : x_offset + width] = opencv_img
        return canvas


def generate_all(
    object_longest_side,
    linedrawing_input_folder,
    dot_distance,
    dot_size,
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    regenerate=DEFAULTS["regenerate"],
):
    linedrawing_input_folder = (
        Path("assets") / "baker_2018" / "outline_images_fix"
        if linedrawing_input_folder is None
        else Path(linedrawing_input_folder)
    )

    output_folder = (
        Path("data") / "gestalt" / "dotted_linedrawings"
        if output_folder is None
        else Path(output_folder)
    )

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return output_folder

    delete_and_recreate_path(output_folder)

    all_categories = [i.stem for i in linedrawing_input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    ds = DrawDottedImage(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Class", "Background", "DotDistance", "DotSize", "IterNum"]
        )
        for n, img_path in enumerate(linedrawing_input_folder.glob("*")):
            class_name = img_path.stem
            img = ds.dotted_image(
                img_path, dot_distance=dot_distance, dot_size=dot_size
            )

            path = Path(class_name) / f"{n}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, background, dot_distance, dot_size, n])


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
    parser.add_argument(
        "--folder_linedrawings",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default="assets/baker_2018/outline_images_fix/",
    )

    parser.add_argument(
        "--dot_distance", "-dd", default=5, help="Distance between dots", type=int
    )
    parser.add_argument(
        "--dot_size", "-ds", default=1, help="Size of each dot", type=int
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
