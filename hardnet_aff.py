try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import add_common_args, get_dict_from_parser, load_data, train_net

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    name = 'hardnetAff'
    args = get_args(name)
    setproctitle(name+'-{}'.format(args['probType']))
    data = load_data(args, DEVICE)

    save_dir = os.path.join('results', str(data), name+args['suffix'],
        time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    soft_epochs = args['softEpochs']
    def modify_net(net, epoch):
        """turn off projection during the initial soft-constrained learning"""
        if epoch < soft_epochs:
            net.set_projection(False)
        else:
            net.set_projection(True)
        return net

    train_net(data, args, HardNetAff, save_dir, net_modifier_fn=modify_net)


def get_args(name):
    parser = argparse.ArgumentParser(description=name)
    parser = add_common_args(parser)
    parser.add_argument('--softEpochs', type=int,
        help='# of initial epochs for warm start to do soft-constrained learning')
    args = get_dict_from_parser(parser, name)
    print(f'{name}: {args}')
    return args


######### Models

class HardNetAff(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        self._if_project = False
        layer_sizes = [data.encoded_xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        output_dim = data.ydim - data.neq - data.nknowns

        if self._args['probType'] == 'nonconvex': # follow DC3 paper's setting for reproducing its results
            layers = reduce(operator.add,
                [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                    for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        else:
            layers = reduce(operator.add,
                [[nn.Linear(a,b), nn.ReLU()]
                    for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self._net = nn.Sequential(*layers)
    
    def set_projection(self, val=True):
        """set wether to do projection or not"""
        self._if_project = val

    def apply_projection(self, f, A, b):
        """project f to satisfy Af<=b"""
        if self._args['probType'] == 'nonconvex': # efficient computation for input-independent A
            A = A[0,:,:]
            return f - (torch.linalg.pinv(A) @ nn.ReLU()(A @ f[:,:,None] - b[:,:,None]))[:,:,0]
        
        # return f - (torch.linalg.pinv(A) @ nn.ReLU()(A @ f[:,:,None] - b[:,:,None]))[:,:,0]
        # listsq is more stable than pinv
        return f - torch.linalg.lstsq(A, nn.ReLU()(A @ f[:,:,None] - b[:,:,None])).solution[:,:,0]

    def forward(self, x, isTest=False):
        encoded_x = self._data.encode_input(x)
        out = self._net(encoded_x)

        if self._if_project:
            A_eff, b_eff = self._data.get_Ab_effective(x)
            out = self.apply_projection(out, A_eff, b_eff)
        
        return self._data.complete_partial(x, out)

if __name__=='__main__':
    main()