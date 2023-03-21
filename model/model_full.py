import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from utils.preprocess_input_pytorch import DataItem

from .model_subnetwork1 import DefaultModel as Subnetwork1
from .model_subnetwork2 import DefaultModel as Subnetwork2
from .model_subnetwork3 import DefaultModel as Subnetwork3


class DefaultModel(BaseModel):
    """
        Full model
    """

    def __init__(self, init_dim=32, norm_type='instance', C=3):
        super(DefaultModel, self).__init__()

        self.Subnetwork1 = Subnetwork1(init_dim, norm_type, False, C)
        self.Subnetwork2 = Subnetwork2(init_dim, norm_type, True, C)
        self.Subnetwork3 = Subnetwork3(init_dim, norm_type, True, C)

    def forward(self, L_cat):
        # |input:
        #  L_cat: three polarized images, [0, 1], as float32
        # |output:
        #  I_cat_pred: four dequantized polarized LDR images (concatenated), [0, 1], as float32
        #  p_pred: restored value of the degree of polarization, [0, 1] float, as float32
        #  theta_pred: restored value of the angle of polarization, [0, 1] float, as float32
        #  H_pred: reconstructed unpolarized HDR image, [0, 1+] float, as float32

        data_item = DataItem(L_cat)

        L_cat = data_item.get_net1_input()
        I_cat_pred = self.Subnetwork1(L_cat).detach()  # freeze Subnetwork1
        data_item.set_net1_output(I_cat_pred)

        I_cat_pred, p_hat_pred, theta_hat_pred, p_and_theta_weight_pred = data_item.get_net2_input()
        p_pred, theta_pred = self.Subnetwork2(I_cat_pred, p_hat_pred, theta_hat_pred, p_and_theta_weight_pred)
        data_item.set_net2_output(p_pred, theta_pred)

        H_hat_2_pred, H_weight_pred = data_item.get_net3_input()
        H_pred = self.Subnetwork3(p_pred, theta_pred, H_hat_2_pred, H_weight_pred)
        data_item.set_net3_output(H_pred)

        return I_cat_pred, p_pred, theta_pred, H_pred