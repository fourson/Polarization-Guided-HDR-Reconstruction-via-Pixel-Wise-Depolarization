import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning H (HDR reconstruction)

        as input:
        p: degree of polarization, [0, 1] float, as float32
        theta: angle of polarization, [0, 1] float, as float32
        H_hat_2: updated value of the unpolarized HDR image, [0, 1+], as float32
        H_weight: weight function, [0, 1] float, as float32

        as target:
        H: unpolarized HDR image, [0, 1+] float, as float32
    """

    def __init__(self, data_dir, transform=None):
        self.p_dir = os.path.join(data_dir, 'share', 'p')
        self.theta_dir = os.path.join(data_dir, 'share', 'theta')
        self.H_hat_2_dir = os.path.join(data_dir, 'subnetwork3', 'H_hat_2')
        self.H_weight_dir = os.path.join(data_dir, 'subnetwork3', 'H_weight')

        self.H_dir = os.path.join(data_dir, 'subnetwork3', 'H')

        self.names = fnmatch.filter(os.listdir(self.H_hat_2_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, C)
        p = np.load(os.path.join(self.p_dir, self.names[index]))  # [0, 1] float, as float32
        theta = np.load(os.path.join(self.theta_dir, self.names[index])) / np.pi  # [0, 1] float, as float32
        H_hat_2 = np.load(os.path.join(self.H_hat_2_dir, self.names[index]))  # [0, 1+] float, as float32
        H_weight = np.load(os.path.join(self.H_weight_dir, self.names[index]))  # [0, 1] float, as float32

        # as target:
        # (H, W, C)
        H = np.load(os.path.join(self.H_dir, self.names[index]))  # [0, 1+] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            p = self.transform(p)
            theta = self.transform(theta)
            H_hat_2 = self.transform(H_hat_2)
            H_weight = self.transform(H_weight)

            H = self.transform(H)

        return {'p': p, 'theta': theta, 'H_hat_2': H_hat_2, 'H_weight': H_weight, 'H': H, 'name': name}


class InferDataset(Dataset):
    """
        for learning H (HDR reconstruction)

        as input:
        p: degree of polarization, [0, 1] float, as float32
        theta: angle of polarization, [0, 1] float, as float32
        H_hat_2: updated value of the unpolarized HDR image, [0, 1+], as float32
        H_weight: weight function, [0, 1] float, as float32
    """

    def __init__(self, data_dir, transform=None):
        self.p_dir = os.path.join(data_dir, 'share', 'p')
        self.theta_dir = os.path.join(data_dir, 'share', 'theta')
        self.H_hat_2_dir = os.path.join(data_dir, 'subnetwork3', 'H_hat_2')
        self.H_weight_dir = os.path.join(data_dir, 'subnetwork3', 'H_weight')

        self.names = fnmatch.filter(os.listdir(self.H_hat_2_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, C)
        p = np.load(os.path.join(self.p_dir, self.names[index]))  # [0, 1] float, as float32
        theta = np.load(os.path.join(self.theta_dir, self.names[index])) / np.pi  # [0, 1] float, as float32
        H_hat_2 = np.load(os.path.join(self.H_hat_2_dir, self.names[index]))  # [0, 1+] float, as float32
        H_weight = np.load(os.path.join(self.H_weight_dir, self.names[index]))  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            p = self.transform(p)
            theta = self.transform(theta)
            H_hat_2 = self.transform(H_hat_2)
            H_weight = self.transform(H_weight)

        return {'p': p, 'theta': theta, 'H_hat_2': H_hat_2, 'H_weight': H_weight, 'name': name}
