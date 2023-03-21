import os
import argparse
import importlib
import sys

import torch
from tqdm import tqdm
import numpy as np


def infer_default():
    H_pred_dir = os.path.join(result_dir, 'H_pred')
    util.ensure_dir(H_pred_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]

            # get data and send them to GPU
            # (N, C, H, W) GPU tensor
            p = sample['p'].to(device)
            theta = sample['theta'].to(device)
            H_hat_2 = sample['H_hat_2'].to(device)
            H_weight = sample['H_weight'].to(device)

            # get network output
            # (N, C, H, W) GPU tensor
            H_pred = model(p, theta, H_hat_2, H_weight)

            # save data
            H_pred_numpy = np.transpose(H_pred.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(H_pred_dir, name + '.npy'), H_pred_numpy)


if __name__ == '__main__':
    MODULE = 'subnetwork3'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='InferDataLoader', type=str, help='which data loader to use')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser = subparsers.add_parser("default")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')
    data_loader_class = getattr(module_data, args.data_loader_type)
    data_loader = data_loader_class(data_dir=args.data_dir)

    # build model architecture
    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # run the selected func
    if args.func == 'default':
        infer_default()
    else:
        # run the default
        infer_default()
