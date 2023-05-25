import argparse
import glob
import os.path
import shutil
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.compute_distance.misc import (
    paste_at_center,
    my_affine,
    get_new_affine_values,
)
from src.utils.drawing_utils import DrawShape, get_mask_from_linedrawing
from src.utils.misc import add_general_args


# def generate_training_dataset():
class DrawTrainingImages(DrawShape):
    def __init__(self, obj_size_ratio, transform_code, *args, **kwargs):
        self.obj_size_ratio = obj_size_ratio
        self.transform_code = transform_code
        self.obj_size = tuple(
            (np.array(self.obj_size_ratio) * self.canvas_size).astype(int)
        )
        super().__init__(*args, **kwargs)

    def create_training_image(self, img_path):
        expand_factor = 3
        mask = get_mask_from_linedrawing(img_path, self.obj_size)
        original_canvas_size = self.canvas_size
        self.canvas_size = tuple(
            (np.array(self.canvas_size) * expand_factor).astype(int)
        )
        # self.canvas_size = tuple((np.array(self.canvas_size) * 1.5).astype(int))

        canvas = self.create_canvas()
        canvas_foreg_text = self.create_canvas(
            size=tuple(
                (np.array(self.obj_size_ratio) * original_canvas_size).astype(int)
            ),
            background=self.line_args["fill"],
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        self.canvas_size = original_canvas_size

        af = get_new_affine_values(self.transform_code)
        img = my_affine(
            canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        img = transforms.CenterCrop(self.canvas_size[0])(img)
        return img


def generate_all(
    linedrawing_input_folder,
    output_folder,
    canvas_size,
    background,
    antialiasing,
    object_size,
    num_samples,
    transform_code="r[-180,180]t[-0.2,0.2]s[0.5,1.2]",
):
    output_folder = Path(output_folder)
    linedrawing_input_folder = Path(linedrawing_input_folder)
    [
        shutil.rmtree(f)
        for f in glob.glob(str(output_folder / "**"))
        if os.path.exists(f)
    ]
    [
        Path(output_folder / os.path.splitext(os.path.basename(f))[0]).mkdir(
            exist_ok=True, parents=True
        )
        for f in glob.glob(str(linedrawing_input_folder / "**"))
    ]

    ds = DrawTrainingImages(
        background=background,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_size_ratio=np.array(object_size) / np.array(canvas_size),
        transform_code=transform_code,
    )
    for f in glob.glob(str(linedrawing_input_folder / "**")):
        class_name = os.path.splitext(os.path.basename(f))[0]
        print(class_name)
        for n in range(num_samples):
            img = ds.create_training_image(f)
            img.save(output_folder / class_name / f"{n}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    parser.add_argument(
        "--object_size",
        "-objsize",
        default="100x100",
        help="A string in the format NxM specifying the size of object (in pixels) embedded into the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")]),
    )
    parser.add_argument(
        "--num_samples",
        "-ns",
        default=10000,
        help="The number of augmented samples to generate for each line drawings",
        type=int,
    )
    parser.add_argument(
        "--folder_linedrawings",
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
