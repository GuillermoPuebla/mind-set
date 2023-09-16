"""
Script for regression test / revisit doc string later
"""
from src.utils.generate_default_pars_toml_file import create_config
from src.generate_datasets_from_toml import generate_toml
from pathlib import Path
import toml
import shutil
import os
from collections import defaultdict
import base64
from nbformat import v4 as nbf


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
    return encoded_bytes.decode("utf-8")


def build_dataset_structure(path, dataset_structure):
    # collect first 9
    for root, dirs, files in sorted(os.walk(path)):
        image_files = [
            Path(root) / file for file in sorted(files) if file.endswith(".png")
        ][:9]
        if image_files:
            d = dataset_structure
            path_parts = Path(root).relative_to(path).parts
            for part in path_parts:
                d = d.setdefault(part, defaultdict(dict))
            d["images"] = image_files


def generate_headers(nb, data, level=1):
    for key, value in sorted(data.items()):
        if key != "images":
            nb.cells.append(nbf.new_markdown_cell(f"{'#' * level} {key}"))
            generate_headers(nb, value, level + 1)
        else:
            nb.cells.append(nbf.new_markdown_cell(create_table_markdown(value)))


def create_table_markdown(images):
    table_markdown = "<table><tr>"
    for index, image_path in enumerate(images):
        base64_data = encode_image_base64(image_path)
        if base64_data:
            table_markdown += f"<td><img src='data:image/png;base64,{base64_data}' alt='{image_path.name}'></td>"
            if (index + 1) % 3 == 0 or (index + 1) == len(images):
                table_markdown += "</tr><tr>"
    table_markdown += "</tr></table>"
    return table_markdown


def generate_notebook(dataset_structure):
    save_to = Path("tests", "samples_preview.ipynb")
    nb = nbf.new_notebook()
    generate_headers(nb, dataset_structure)
    with open(save_to, "w") as f:
        f.write(nbf.writes(nb))


def test_generate_toml():
    # -------- folder preparation --------
    toml_save_to = Path("tests", "all_datasets.toml")
    data_save_to = Path("tests", "sample_data")
    toml_individual_save_to = Path("tests", "tomls")
    toml_individual_save_to.mkdir(parents=True, exist_ok=True)
    if data_save_to.exists():
        shutil.rmtree(data_save_to)

    # -------- making config --------
    create_config(save_to=toml_save_to)

    with open(toml_save_to, "r") as f:
        toml_config = toml.load(f)

    name_path_dict = {}

    for key in toml_config:
        dataset_name = Path(key).parent.name
        dataset_category_name = Path(key).parent.parent.name
        output_folder = data_save_to / dataset_category_name / dataset_name
        toml_config[key].update({"output_folder": output_folder.as_posix()})
        name_path_dict.update({dataset_name: output_folder})

        # -------- set num_samples to 10 --------
        for sub_key in toml_config[key]:
            if "num_samples" in sub_key:
                toml_config[key][sub_key] = 9

        with open(toml_individual_save_to / f"{dataset_name}.toml", "w") as f:
            toml.dump({key: toml_config[key]}, f)

    # -------- generate each dataset --------
    [generate_toml(toml_individual_save_to / f"{key}.toml") for key in name_path_dict]

    # -------- collect generated images --------
    dataset_structure = defaultdict(dict)
    build_dataset_structure(data_save_to, dataset_structure)
    generate_notebook(dataset_structure)


if __name__ == "__main__":
    test_generate_toml()
