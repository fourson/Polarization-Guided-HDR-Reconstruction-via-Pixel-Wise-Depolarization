import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from .networks import get_norm_layer, UnetBackbone


class DefaultModel(BaseModel):
    """
        dequantization and denoise
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3):
        super(DefaultModel, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(4 * C, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.backbone = UnetBackbone(init_dim, output_nc=init_dim, n_downsampling=4, use_conv_to_downsample=False,
                                     norm_type=norm_type, use_dropout=use_dropout, mode='default')

        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, 4 * C, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, L_cat):
        # |input:
        #  L_cat: four quantized polarized LDR images (concatenated), [0, 1], as float32
        # |output:
        #  I_cat_pred: four dequantized polarized LDR images (concatenated), [0, 1], as float32

        feature = self.feature_extraction(L_cat)
        backbone_out = self.backbone(feature)
        I_cat_pred = self.out_block(backbone_out) + L_cat

        I_cat_pred = torch.clamp(I_cat_pred, min=0, max=1)

        return I_cat_pred
