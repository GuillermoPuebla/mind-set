"""
Script for regression test / revisit doc string later
"""

from src.utils.generate_default_pars_toml_file import create_config
from src.generate_datasets_from_toml import generate_toml
from pathlib import Path
import toml
import shutil
from tqdm import tqdm
import multiprocessing as mp
import base64
import nbformat as nbf


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
    return encoded_bytes.decode("utf-8")


class CountImagesIn:
    def __init__(self, path, generate_thread):
        self.limit = 10
        self.path = path
        self.generate_thread = generate_thread
        self.count = self()
        self.progress_bar = tqdm(total=self.limit)
        self.update_loop()

    def __call__(self):
        self.count = sum(1 for _ in self.path.rglob("*.png"))

    def update_loop(self):
        while True:
            self()
            self.progress_bar.update(self.count - self.progress_bar.n)
            if self.count >= self.limit:
                self.generate_thread.terminate()
                break


def test_generate_toml():
    toml_save_to = Path("tests", "all_datasets.toml")
    data_save_to = Path("tests", "sample_data")
    toml_individual_save_to = Path("tests", "tomls")
    toml_individual_save_to.mkdir(parents=True, exist_ok=True)
    if data_save_to.exists():
        shutil.rmtree(data_save_to)

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

        with open(toml_individual_save_to / f"{dataset_name}.toml", "w") as f:
            toml.dump({key: toml_config[key]}, f)

    # -------- generate each dataset --------
    for key in name_path_dict:
        toml_path = toml_individual_save_to / f"{key}.toml"
        output_path = name_path_dict[key]

        process_generate = mp.Process(target=generate_toml, args=(toml_path,))
        process_generate.start()
        CountImagesIn(output_path, generate_thread=process_generate)

    # -------- collect generated images --------
    dataset_structure = {}
    for key in toml_config:
        dataset_name = Path(key).parent.name
        dataset_category_name = Path(key).parent.parent.name
        output_folder = data_save_to / dataset_category_name / dataset_name
        images = list(output_folder.rglob("*.png"))

        if dataset_category_name not in dataset_structure:
            dataset_structure[dataset_category_name] = {}
        if dataset_name not in dataset_structure[dataset_category_name]:
            dataset_structure[dataset_category_name][dataset_name] = []
        dataset_structure[dataset_category_name][dataset_name].extend(images)

    nb = nbf.v4.new_notebook()

    for category_name, datasets in dataset_structure.items():
        nb.cells.append(nbf.v4.new_markdown_cell(f"# {category_name}"))

        for dataset_name, images in datasets.items():
            nb.cells.append(nbf.v4.new_markdown_cell(f"## {dataset_name}"))

            # Initialize table markdown
            table_markdown = "<table><tr>"

            for index, image_path in enumerate(images):
                base64_data = encode_image_base64(image_path)

                if len(base64_data):  # filter empty
                    table_markdown += f"<td><img src='data:image/png;base64,{base64_data}' alt='{image_path.name}'></td>"
                    # check last
                    if (index + 1) % 3 == 0 or (index + 1) == len(images):
                        table_markdown += "</tr><tr>"

            # close table
            table_markdown += "</tr></table>"
            nb.cells.append(nbf.v4.new_markdown_cell(table_markdown))

    with open("GeneratedNotebook.ipynb", "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    test_generate_toml()
