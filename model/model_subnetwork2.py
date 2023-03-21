import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from .networks import get_norm_layer, AutoencoderBackbone


class DefaultModel(BaseModel):
    """
        DoP and AoP restoration
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction1 = nn.Sequential(
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
        self.feature_extraction2 = nn.Sequential(
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
        self.feature_extraction3 = nn.Sequential(
            nn.Conv2d(4 * C, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.feature_extraction4 = nn.Sequential(
            nn.Conv2d(4 * C, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim // 2, init_dim // 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.backbone1 = AutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=3, n_blocks=5, norm_type=norm_type,
                                        use_dropout=use_dropout)
        self.out_block1 = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.backbone2 = AutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=3, n_blocks=5, norm_type=norm_type,
                                        use_dropout=use_dropout)
        self.out_block2 = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, I_cat, p_hat, theta_hat, p_and_theta_weight):
        # |input:
        #  I_cat: four unquantized polarized LDR images (concatenated), [0, 1], as float32
        #  p_hat: coarse value of the degree of polarization, [0, 1], as float32
        #  theta_hat: coarse value of the angle of polarization, [0, 1] float, as float32
        #  p_and_theta_weight: weight function, [0, 1] float, as float32
        # |output:
        #  p_pred: restored value of the degree of polarization, [0, 1] float, as float32
        #  theta_pred: restored value of the angle of polarization, [0, 1] float, as float32

        feature1 = self.feature_extraction1(p_hat)
        feature2 = self.feature_extraction2(theta_hat)
        feature3 = self.feature_extraction3(I_cat)
        feature4 = self.feature_extraction4(I_cat)

        backbone_out1 = self.backbone1(torch.cat([feature1, feature3], dim=1))
        p_pred = self.out_block1(backbone_out1) * p_and_theta_weight + p_hat
        p_pred = torch.clamp(p_pred, min=0, max=1)

        backbone_out2 = self.backbone2(torch.cat([feature2, feature4], dim=1))
        theta_pred = self.out_block2(backbone_out2) * p_and_theta_weight + theta_hat
        theta_pred = torch.clamp(theta_pred, min=0, max=1)

        return p_pred, theta_pred
