import PIL.Image as Image
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import List, Optional, Union, Dict
import pathlib


class ImageDatasetAnnotations(Dataset):
    def __init__(
        self,
        task_type: str,
        csv_file: str,
        img_path_col: str,
        label_cols: Union[List[str], str],
        filters: Optional[Dict[str, Union[str, int]]] = None,
        transform=None,
    ):
        self.task_type = task_type
        self.dataframe = pd.read_csv(csv_file)
        if filters:
            for key, value in filters.items():
                self.dataframe = self.dataframe[self.dataframe[key] == value]
        self.img_path_col = img_path_col
        self.root_path = pathlib.Path(csv_file).parent
        self.label_cols = label_cols

        if isinstance(self.label_cols, str):
            self.label_cols = [self.label_cols]

        if self.task_type == "classification":
            assert (
                len(self.label_cols) == 1
            ), "With a classification task, the dataset.label_cols must be a single string or one-element list"
            self.label_cols = self.label_cols[0]
            self.classes = self.dataframe[self.label_cols].unique()

        self.transform = transform
        self.dataframe = self.dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.root_path / self.dataframe.loc[idx, self.img_path_col]
        if self.task_type == "classification":
            labels = self.dataframe.loc[idx, self.label_cols]
            label_tensor_dtype = torch.long
        else:
            labels = self.dataframe.loc[idx, self.label_cols].values.astype(float)
            label_tensor_dtype = torch.float32
        image: Image.Image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=label_tensor_dtype)
