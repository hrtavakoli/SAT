import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import resnetsal

# Prune settings
parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data', type=str, default='/scratch/zhuangl/datasets/imagenet',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# if not os.path.exists(args.save):
#    os.makedirs(args.save)


model = resnetsal.Model().cuda()
checkpoint = torch.load('./pre_train/resnetsal/model_best_256x320.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
print('Pre-processing Successful!')


prune_prob = 0.1
prune_prob = 0.25

layer_id = 1

Layers_mask = []
for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        out_channels = m.weight.data.shape[0]
        prune_prob_stage = prune_prob
        weight_copy = m.weight.data.abs().clone().cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
        num_prune = int(out_channels * prune_prob_stage)
        arg_max = np.argsort(L1_norm)
        arg_max_prune = arg_max[:num_prune]
        mask = torch.zeros(out_channels)
        mask[arg_max_prune.tolist()] = 1.0
        idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        Layers_mask.append(idx)
        m.weight.data[idx.tolist(),:,:,:] = 0.0


torch.save({'state_dict': model.state_dict(),
            'prune_mask': Layers_mask}, os.path.join('p_{}_pruned.pth.tar'.format(prune_prob)))

