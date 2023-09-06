import PIL.Image as Image
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import List, Optional, Union, Dict
import pathlib

a= ['prova']


class ImageRegressionDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_path_col: str,
        label_cols: List[str],
        filters: Optional[Dict[str, Union[str, int]]] = None,
        transform=None,
    ):
        self.dataframe = pd.read_csv(csv_file)
        if filters:
            for key, value in filters.items():
                self.dataframe = self.dataframe[self.dataframe[key] == value]
        self.img_path_col = img_path_col
        self.root_path = pathlib.Path(csv_file).parent
        self.label_cols = label_cols
        self.transform = transform
        self.dataframe = self.dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.root_path / self.dataframe.loc[idx, self.img_path_col]
        labels = self.dataframe.loc[idx, self.label_cols].values.astype(float)
        image: Image.Image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)
