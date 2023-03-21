import os

import torch
import torch.nn.functional as F
import cv2
import numpy as np

TonemapReinhard = cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)
Laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).cuda()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork1':
        # 400ep
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork2':
        # 400ep
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork3':
        # 400ep
        return lambda epoch: 1
    elif lr_lambda_tag == 'full':
        # 300ep
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


def tonemap(hdr_tensor):
    # tonemap hdr image tensor(N, C, H, W) for visualization when using tensorboard
    tonemapped_tensor = torch.zeros(hdr_tensor.shape, dtype=torch.float32)
    for i in range(hdr_tensor.shape[0]):
        hdr = hdr_tensor[i].numpy().transpose((1, 2, 0))  # (H, W, C)
        is_rgb = (hdr.shape[2] == 3)
        if is_rgb:
            # if RGB (H, W, 3) , we should convert to an (H, W, 3) numpy array in order of BGR before tonemapping
            hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
        else:
            # if grayscale (H ,W, 1), we should copy the image 3 times to an (H, W, 3) numpy array before tonemapping
            hdr = cv2.merge([hdr, hdr, hdr])
        hdr = (hdr - np.min(hdr)) / (np.max(hdr) - np.min(hdr))
        tonemapped = TonemapReinhard.process(hdr)
        if is_rgb:
            # back to (C, H, W) tensor in order of RGB
            tonemapped_tensor[i] = torch.from_numpy(cv2.cvtColor(tonemapped, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        else:
            tonemapped_tensor[i] = torch.from_numpy(tonemapped[:, :, 0:1].transpose((2, 0, 1)))
    return tonemapped_tensor


def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    pad = [1, 1, 1, 1]
    laplacian_kernel = Laplacian.view(1, 1, 3, 3)
    edge_tensor = torch.zeros(img_tensor.shape, dtype=torch.float32).cuda()
    for i in range(img_tensor.shape[1]):
        padded = F.pad(img_tensor[:, i:i + 1, :, :], pad, mode='reflect')
        edge_tensor[:, i:i + 1, :, :] = F.conv2d(padded, laplacian_kernel)
    return edge_tensor
