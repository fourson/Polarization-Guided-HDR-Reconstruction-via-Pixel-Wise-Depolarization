import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from .networks import get_norm_layer, UnetBackbone


class DefaultModel(BaseModel):
    """
        HDR reconstruction
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=True, C=3):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction1 = nn.Sequential(
            nn.Conv2d(C, init_dim // 4, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 4, init_dim // 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 4, init_dim // 4, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.feature_extraction2 = nn.Sequential(
            nn.Conv2d(C, init_dim // 4, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 4, init_dim // 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 4, init_dim // 4, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.feature_extraction3 = nn.Sequential(
            nn.Conv2d(C, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.backbone = UnetBackbone(init_dim, output_nc=init_dim, n_downsampling=4, use_conv_to_downsample=False,
                                     norm_type=norm_type, use_dropout=use_dropout, mode='default')

        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, p, theta, H_hat_2, H_weight):
        # |input:
        #  p: degree of polarization, [0, 1] float, as float32
        #  theta: angle of polarization, [0, 1] float, as float32
        #  H_hat_2: updated value of the unpolarized HDR image, [0, 1+], as float32
        #  H_weight: weight function, [0, 1] float, as float32
        # |output:
        #  H_pred: reconstructed unpolarized HDR image, [0, 1+] float, as float32

        feature1 = self.feature_extraction1(p)
        feature2 = self.feature_extraction2(theta)
        feature3 = self.feature_extraction3(H_hat_2)
        backbone_out = self.backbone(torch.cat([feature1, feature2, feature3], dim=1))
        H_pred = self.out_block(backbone_out) * H_weight + H_hat_2
        H_pred = torch.clamp(H_pred, min=0)
        return H_pred
