import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning p and theta (DoP and AoP restoration)

        as input:
        I_cat: four unquantized polarized LDR images (concatenated), [0, 1], as float32
        p_hat: coarse value of the degree of polarization, [0, 1], as float32
        theta_hat: coarse value of the angle of polarization, [0, 1] float, as float32
        p_and_theta_weight: weight function, [0, 1] float, as float32

        as target:
        p: degree of polarization, [0, 1] float, as float32
        theta: angle of polarization, [0, 1] float, as float32
    """

    def __init__(self, data_dir, transform=None):
        self.I_cat_dir = os.path.join(data_dir, 'share', 'I_cat')
        self.p_hat_dir = os.path.join(data_dir, 'subnetwork2', 'p_hat')
        self.theta_hat_dir = os.path.join(data_dir, 'subnetwork2', 'theta_hat')
        self.p_and_theta_weight_dir = os.path.join(data_dir, 'subnetwork2', 'p_and_theta_weight')

        self.p_dir = os.path.join(data_dir, 'share', 'p')
        self.theta_dir = os.path.join(data_dir, 'share', 'theta')

        self.names = fnmatch.filter(os.listdir(self.I_cat_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 4*C)
        I_cat = np.load(os.path.join(self.I_cat_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        p_hat = np.load(os.path.join(self.p_hat_dir, self.names[index]))  # [0, 1] float, as float32
        theta_hat = np.load(os.path.join(self.theta_hat_dir, self.names[index])) / np.pi  # [0, 1] float, as float32
        p_and_theta_weight = np.load(
            os.path.join(self.p_and_theta_weight_dir, self.names[index]))  # [0, 1] float, as float32

        # as target:
        # (H, W, C)
        p = np.load(os.path.join(self.p_dir, self.names[index]))  # [0, 1] float, as float32
        theta = np.load(os.path.join(self.theta_dir, self.names[index])) / np.pi  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_cat = self.transform(I_cat)
            p_hat = self.transform(p_hat)
            theta_hat = self.transform(theta_hat)
            p_and_theta_weight = self.transform(p_and_theta_weight)

            p = self.transform(p)
            theta = self.transform(theta)

        return {'I_cat': I_cat, 'p_hat': p_hat, 'theta_hat': theta_hat, 'p_and_theta_weight': p_and_theta_weight,
                'p': p, 'theta': theta, 'name': name}


class InferDataset(Dataset):
    """
        for learning p and theta (DoP and AoP restoration)

        as input:
        I_cat: four unquantized polarized LDR images (concatenated), [0, 1], as float32
        p_hat: coarse value of the degree of polarization, [0, 1], as float32
        theta_hat: coarse value of the angle of polarization, [0, 1] float, as float32
        p_and_theta_weight: weight function, [0, 1] float, as float32
    """

    def __init__(self, data_dir, transform=None):
        self.I_cat_dir = os.path.join(data_dir, 'share', 'I_cat')
        self.p_hat_dir = os.path.join(data_dir, 'subnetwork2', 'p_hat')
        self.theta_hat_dir = os.path.join(data_dir, 'subnetwork2', 'theta_hat')
        self.p_and_theta_weight_dir = os.path.join(data_dir, 'subnetwork2', 'p_and_theta_weight')

        self.names = fnmatch.filter(os.listdir(self.I_cat_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 4*C)
        I_cat = np.load(os.path.join(self.I_cat_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        p_hat = np.load(os.path.join(self.p_hat_dir, self.names[index]))  # [0, 1] float, as float32
        theta_hat = np.load(os.path.join(self.theta_hat_dir, self.names[index])) / np.pi  # [0, 1] float, as float32
        p_and_theta_weight = np.load(
            os.path.join(self.p_and_theta_weight_dir, self.names[index]))  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_cat = self.transform(I_cat)
            p_hat = self.transform(p_hat)
            theta_hat = self.transform(theta_hat)
            p_and_theta_weight = self.transform(p_and_theta_weight)

        return {'I_cat': I_cat, 'p_hat': p_hat, 'theta_hat': theta_hat, 'p_and_theta_weight': p_and_theta_weight,
                'name': name}
