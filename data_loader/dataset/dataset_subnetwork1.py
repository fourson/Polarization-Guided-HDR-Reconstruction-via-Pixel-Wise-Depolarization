import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning I1, I2, I3, I4 (dequantization and denoise)

        as input:
        L_cat: four quantized polarized LDR images (concatenated), [0, 1], as float32

        as target:
        I_cat: four unquantized polarized LDR images (concatenated), [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L_cat_dir = os.path.join(data_dir, 'subnetwork1', 'L_cat')

        self.I_cat_dir = os.path.join(data_dir, 'share', 'I_cat')

        self.names = fnmatch.filter(os.listdir(self.L_cat_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 4*C)
        L_cat = np.load(os.path.join(self.L_cat_dir, self.names[index]))  # [0, 1] float, as float32

        # as target:
        # (H, W, 4*C)
        I_cat = np.load(os.path.join(self.I_cat_dir, self.names[index]))  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            L_cat = self.transform(L_cat)

            I_cat = self.transform(I_cat)

        return {'L_cat': L_cat, 'I_cat': I_cat, 'name': name}


class InferDataset(Dataset):
    """
        for learning I1, I2, I3, I4 (dequantization and denoise)

        as input:
        L_cat: four quantized polarized LDR images (concatenated), [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L_cat_dir = os.path.join(data_dir, 'subnetwork1', 'L_cat')

        self.names = fnmatch.filter(os.listdir(self.L_cat_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 4*C)
        L_cat = np.load(os.path.join(self.L_cat_dir, self.names[index]))  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            L_cat = self.transform(L_cat)

        return {'L_cat': L_cat, 'name': name}
