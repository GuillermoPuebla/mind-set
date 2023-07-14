import argparse
import csv
import glob
import os
import pathlib
import shutil
import PIL.Image as Image
import numpy as np
import sty
from torchvision import transforms

from src.datasets_generation.gestalt.thatcher_illusion.utils import (
    get_image_facial_landmarks,
    get_bounding_rectangle,
    apply_thatcher_effect_on_image,
)
from src.utils.misc import (
    add_general_args,
    add_training_args,
    delete_and_recreate_path,
    DEFAULTS,
)


def generate_all(
    face_folder,
    output_folder=DEFAULTS["output_foder"],
    canvas_size=DEFAULTS["canvas_size"],
    regenerate=DEFAULTS["regenerate"],
):
    face_folder = (
        pathlib.Path("assets") / "celebA_sample" / "normal"
        if face_folder is None
        else face_folder
    )
    output_folder = (
        pathlib.Path("data") / "gestalt" / "thatcher_illusion"
        if output_folder is None
        else pathlib.Path(output_folder)
    )

    if output_folder.exists() and not regenerate:
        print(
            sty.fg.yellow
            + f"Dataset already exists and regenerate if false. Finished"
            + sty.rs.fg
        )
        return output_folder

    delete_and_recreate_path(output_folder)
    [
        (output_folder / cond).mkdir(parents=True, exist_ok=True)
        for cond in ["normal", "inverted", "thatcherized"]
    ]
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Transformation", "FaceId"])
        for idx, f in enumerate(face_folder.glob("*")):
            image_facial_landmarks = get_image_facial_landmarks(f)
            if (
                not image_facial_landmarks
                or len(image_facial_landmarks) == 0
                or len(image_facial_landmarks) != 68
            ):
                continue
            left_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[36:42])
            right_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[42:48])
            mouth_rectangle = get_bounding_rectangle(image_facial_landmarks[48:68])
            cv_image = apply_thatcher_effect_on_image(
                str(f),
                np.array(left_eye_rectangle).astype(int),
                np.array(right_eye_rectangle).astype(int),
                np.array(mouth_rectangle).astype(int),
            )
            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.fromarray(cv_image)
            ).save(output_folder / "thatcherized" / f"{idx}.png")
            writer.writerow(
                [pathlib.Path("thatcherized") / f"{idx}.png", "thatcherized", idx]
            )

            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(Image.open(f)).save(
                output_folder / "normal" / f"{idx}.png"
            )
            writer.writerow([pathlib.Path("normal") / f"{idx}.png", "normal", idx])

            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.open(f).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            ).save(output_folder / "inverted" / f"{idx}.png")
            writer.writerow([pathlib.Path("inverted") / f"{idx}.png", "inverted", idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        default=None,
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure used to create the dataset",
    )

    parser.add_argument(
        "--face_folder",
        "-ff",
        default=None,
        help="The folder containing faces that need to be Thatcherized. These faces will also be resized to `canvas_size` size. ",
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default="224x224",
        help="A string in the format NxM specifying the size of the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")]),
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
