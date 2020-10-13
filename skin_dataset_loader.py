import os
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset


class SkinLesion(Dataset):
    category2num_label_converting = {"nevus":0,"melanoma":1,"seborrheic_keratosis":2}

    def __init__(self, df, transforms=None, **kwargs):

        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        img = Image.open(data.pathImg)
            #img = torch.from_numpy(img)
        target = SkinLesion.category2num_label_converting[data.category]
        # yapf: enable
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


