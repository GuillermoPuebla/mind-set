import numpy as np
from pathlib import Path
import shutil


def normalize_labels(labels):
    labels_min = np.min(labels)
    labels_max = np.max(labels)
    return 2 * (labels - labels_min) / (labels_max - labels_min) - 1


def normalize_folder(input_folder: Path):
    """
    the input folder will be a Path object that contains all the images
    where each image is labeled as {label}_{uuid}.png where the label is before the character "_"
    in all instances
    1. all the label values would be read and stored in a np array
    2. the labels would be normalized to be between -1 and 1 for each image
    3. a new folder would be created to store the normalized images at the same location as the input folder
    4. the new folder would be named {input_folder_name}_normalized
    """
    assert input_folder.exists(), f"Input folder {input_folder} does not exist"

    output_folder = input_folder.parent / f"{input_folder.name}_normalized"
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(input_folder.glob("*_*.png"))

    labels = []
    for image_file in image_files:
        label = image_file.stem.split("_")[0]
        labels.append(float(label))
    labels = np.array(labels)

    normalized_labels = normalize_labels(labels)

    for image_file, norm_label in zip(image_files, normalized_labels):
        filename = f"{norm_label:.2f}_{image_file.stem.split('_')[1]}.png"
        output_path = output_folder / filename
        shutil.copy(image_file, output_path)

    print(f"Normalized images saved in {output_folder}")


if __name__ == "__main__":
    input_folder = Path("datasets", "jastrow_train")
    normalize_folder(input_folder)
