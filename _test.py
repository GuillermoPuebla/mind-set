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

    # -------- now generate each dataset --------
    for key in name_path_dict:
        toml_path = toml_individual_save_to / f"{key}.toml"
        output_path = name_path_dict[key]

        process_generate = mp.Process(target=generate_toml, args=(toml_path,))
        process_generate.start()
        CountImagesIn(output_path, generate_thread=process_generate)

    # -------- collect generated images --------
    for key in toml_config:
        dataset_name = Path(key).parent.name
        dataset_category_name = Path(key).parent.parent.name
        output_folder = data_save_to / dataset_category_name / dataset_name

        # to be continued


if __name__ == "__main__":
    test_generate_toml()