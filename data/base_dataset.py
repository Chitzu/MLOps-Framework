import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, csv_file="train.csv", transform=None):
        self.transform = transform
        self.df = pd.read_csv(csv_file)

    def __getitem__(self, index):
        image_path = self.df.loc[index]["file_path"]
        label = self.df.loc[index]["label"]

        image = Image.open(image_path).convert('RGB')
        label = np.array(label, dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image) / 255

        return (image, label)

    def __len__(self):
        return len(self.df)