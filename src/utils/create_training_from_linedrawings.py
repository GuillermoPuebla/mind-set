import argparse
import csv
import os.path
from pathlib import Path

import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.compute_distance.misc import (
    my_affine,
    get_new_affine_values,
)
from src.utils.drawing_utils import (
    DrawStimuli,
    resize_image_keep_aspect_ratio,
    paste_linedrawing_onto_canvas,
)
from src.utils.misc import add_general_args, delete_and_recreate_path


class DrawTrainingImages(DrawStimuli):
    def __init__(self, obj_longest_side, transform_code, *args, **kwargs):
        self.transform_code = transform_code

        super().__init__(*args, **kwargs)

        self.obj_longest_side = obj_longest_side

    def create_training_image(self, img_path):
        expand_factor = 3
        opencv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)
        original_canvas_size = self.canvas_size
        self.canvas_size = tuple(
            (np.array(self.canvas_size) * expand_factor).astype(int)
        )
        img = paste_linedrawing_onto_canvas(
            opencv_img, self.create_canvas(), self.line_args["fill"]
        )

        self.canvas_size = original_canvas_size
        if self.transform_code:
            af = get_new_affine_values(self.transform_code)
            img = my_affine(
                img,
                translate=list(np.array(af["tr"]) / expand_factor),
                angle=af["rt"],
                scale=af["sc"],
                shear=af["sh"],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return img


def generate_all(
    linedrawing_input_folder,
    output_folder,
    canvas_size,
    background,
    antialiasing,
    object_longest_side,
    num_samples,
    transform_code="r[-180,180]t[-0.2,0.2]s[0.5,1.2]",
):
    output_folder = Path(output_folder)
    linedrawing_input_folder = Path(linedrawing_input_folder)
    delete_and_recreate_path(output_folder)
    [
        Path(output_folder / f.stem).mkdir(exist_ok=True, parents=True)
        for f in linedrawing_input_folder.glob("*")
    ]

    ds = DrawTrainingImages(
        background=background,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        transform_code=transform_code,
    )
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Label", "Background"])

        for f in linedrawing_input_folder.glob("*"):
            class_name = f.stem
            print(class_name)
            for n in range(num_samples):
                path = os.path.join(class_name, f"{n}.png")
                img = ds.create_training_image(f)
                img.save(output_folder / path)
                writer.writerow([path, n, ds.background])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=100,
        type=int,
        help="Specify how to resize  the object so that its longest side will be equal to this value, before being pasted into a canvas. The linedrawing will maintain the aspect ratio.",
    )
    parser.add_argument(
        "--num_samples",
        "-ns",
        default=10000,
        help="The number of augmented samples to generate for each line drawings",
        type=int,
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default="assets/baker_2018/outline_images_fix/",
    )
    parser.add_argument(
        "--transform_code",
        "-transfc",
        help="The code for applying the transform",
        default="r[-180,180]t[-0.2,0.2]s[0.5,1.2]",
    )
    parser.set_defaults(output_folder="tmp/")

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


# """"""""""""" Example call """"""""""""""
# create_training.generate_all(
#     str(linedrawing_input_folder),
#     str(output_folder / "train"),
#     canvas_size,
#     background,
#     antialiasing,
#     object_size,
#     num_training_samples,
#     transform_code="r[-180,180]t[-0.2,0.2]s[0.5,1.2]"
#     if num_training_samples > 1
#     else None
#     # we use the default transform code
# )
