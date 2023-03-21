import os
import argparse
import importlib
import sys
import pickle

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
            # (N, 4*C, H, W) GPU tensor
            L_cat = sample['L_cat'].to(device)

            # get network output
            # (N, 4*C, H, W) GPU tensor for I_cat_pred
            # (N, C, H, W) GPU tensor for p_pred, theta_pred, H_pred
            I_cat_pred, p_pred, theta_pred, H_pred = model(L_cat)

            # save data
            H_pred_numpy = np.transpose(H_pred.squeeze().cpu().numpy(), (1, 2, 0))

            np.save(os.path.join(H_pred_dir, name + '.npy'), H_pred_numpy)


def infer_save_all():
    H_pred_dir = os.path.join(result_dir, 'H_pred')
    util.ensure_dir(H_pred_dir)
    data_item_dir = os.path.join(result_dir, 'data_item_pred')
    util.ensure_dir(data_item_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]

            # get data and send them to GPU
            # (N, 4*C, H, W) GPU tensor
            L_cat = sample['L_cat'].to(device)

            # get network output
            # (N, 4*C, H, W) GPU tensor for I_cat_pred
            # (N, C, H, W) GPU tensor for p_pred, theta_pred, H_pred
            I_cat_pred, p_pred, theta_pred, H_pred = model(L_cat)

            # save data
            L_cat_numpy = np.transpose(L_cat.squeeze().cpu().numpy(), (1, 2, 0))
            I_cat_pred_numpy = np.transpose(I_cat_pred.squeeze().cpu().numpy(), (1, 2, 0))
            p_pred_numpy = np.transpose(p_pred.squeeze().cpu().numpy(), (1, 2, 0))
            theta_pred_numpy = np.transpose(theta_pred.squeeze().cpu().numpy(), (1, 2, 0))
            H_pred_numpy = np.transpose(H_pred.squeeze().cpu().numpy(), (1, 2, 0))

            data_item = preprocess_input_numpy.DataItem(L_cat_numpy)
            data_item.set_net1_output(I_cat_pred_numpy)
            data_item.set_net2_output(p_pred_numpy, theta_pred_numpy)
            data_item.set_net3_output(H_pred_numpy)

            f = open(os.path.join(data_item_dir, name + '.pickle'), 'wb')
            pickle.dump(data_item, f, pickle.HIGHEST_PROTOCOL)
            f.close()


if __name__ == '__main__':
    MODULE = 'full'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='InferDataLoader', type=str, help='which data loader to use')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser_default = subparsers.add_parser("default")
    subparser_save_all = subparsers.add_parser("save_all")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util, preprocess_input_numpy

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

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # run the selected func
    if args.func == 'default':
        infer_default()
    elif args.func == 'save_all':
        infer_save_all()
    else:
        # run the default
        infer_default()
