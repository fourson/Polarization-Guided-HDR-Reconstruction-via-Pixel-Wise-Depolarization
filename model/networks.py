import functools

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetDoubleConvBlock(nn.Module):
    """
        Unet double Conv block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetDoubleConvBlock, self).__init__()

        self.mode = mode

        if self.mode == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'bottleneck':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'res-bottleneck':
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
            out_sequence = [
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        if self.mode == 'res-bottleneck':
            x_ = self.projection(x)
            out = self.out_block(x_ + self.bottleneck(x_))
        else:
            out = self.out_block(self.model(x))
        return out


class UnetDownsamplingBlock(nn.Module):
    """
        Unet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_conv, mode='default'):
        super(UnetDownsamplingBlock, self).__init__()

        downsampling_layers = list()
        if use_conv:
            downsampling_layers += [
                nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            downsampling_layers += [nn.MaxPool2d(2)]

        self.model = nn.Sequential(
            nn.Sequential(*downsampling_layers),
            UnetDoubleConvBlock(in_channel, out_channel, norm_layer, use_dropout, use_bias, mode=mode)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UnetUpsamplingBlock(nn.Module):
    """
        Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        out = torch.cat([x2, self.upsample(x1)], dim=1)
        out = self.double_conv(out)
        return out


class UnetBackbone(nn.Module):
    """
        Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=True, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(UnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                UnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


class ResBlock(nn.Module):
    """
        ResBlock using bottleneck structure
        dim -> dim
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out


class AutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=6, norm_type='instance', use_dropout=False):
        super(AutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        ]

        dim = output_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.ReLU(inplace=True)
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            sequence += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(dim // 2),
                nn.ReLU(inplace=True)
            ]
            dim //= 2

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out
