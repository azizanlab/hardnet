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

import pickle
from setproctitle import setproctitle
import os
import time
import argparse

from utils import add_common_args, get_dict_from_parser, load_data, train_net
from hardnet_aff import HardNetAff
from hardnet_cvx import HardNetCvx

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    name = 'baselineDC3'
    args = get_args(name)
    setproctitle(name+'-{}'.format(args['probType']))
    data = load_data(args, DEVICE)

    save_dir = os.path.join('results', str(data), name+args['suffix'], f"seed{args['seed']}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Run method
    train_net(data, args, DC3, save_dir)


def get_args(name):
    parser = argparse.ArgumentParser(description=name)
    parser = add_common_args(parser)
    parser.add_argument('--useCompl', type=bool,
        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=bool,
        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=bool,
        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')
    parser.add_argument('--testProj', choices=['none', 'hardnetaff', 'hardnetcvx'], default='none',
        help='projection to use during testing (for ablation study)')
    args = get_dict_from_parser(parser, name)
    print(f'{name}: {args}')
    return args


######### Models

class DC3(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        self._testProj = args['testProj']
        layer_sizes = [data.encoded_xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        output_dim = data.ydim - data.neq if self._args['useCompl'] else data.ydim
        
        if self._args['probType'] == 'opt': # follow DC3 paper's setting for reproducing its results
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

    def grad_steps(self, X, Y):
        take_grad_steps = self._args['useTrainCorr']
        if take_grad_steps:
            lr = self._args['corrLr']
            num_steps = self._args['corrTrainSteps']
            momentum = self._args['corrMomentum']
            partial_var = self._args['useCompl']
            partial_corr = True if self._args['corrMode'] == 'partial' else False
            if partial_corr and not partial_var:
                assert False, "Partial correction not available without completion."
            Y_new = Y
            old_Y_step = 0
            for i in range(num_steps):
                if partial_corr:
                    Y_step = self._data.get_ineq_partial_grad(X, Y_new)
                else:
                    Y_step = self._data.get_resid_grad(X, Y_new)
                new_Y_step = lr * Y_step + momentum * old_Y_step
                Y_new = Y_new - new_Y_step
                old_Y_step = new_Y_step
            return Y_new
        else:
            return Y
    
    def extra_grad_steps(self, X, Y):
        """Used only at test time, so let PyTorch avoid building the computational graph"""
        take_grad_steps = self._args['useTestCorr']
        if take_grad_steps:
            lr = self._args['corrLr']
            eps_converge = self._args['corrEps']
            max_steps = self._args['corrTestMaxSteps']
            momentum = self._args['corrMomentum']
            partial_var = self._args['useCompl']
            partial_corr = True if self._args['corrMode'] == 'partial' else False
            if partial_corr and not partial_var:
                assert False, "Partial correction not available without completion."
            Y_new = Y
            i = 0
            old_Y_step = 0
            with torch.no_grad():
                while (i == 0 or torch.max(self._data.get_resid(X, Y_new)) > eps_converge) and i < max_steps:
                    if partial_corr:
                        Y_step = self._data.get_ineq_partial_grad(X, Y_new)
                    else:
                        Y_step = self._data.get_resid_grad(X, Y_new)
                    new_Y_step = lr * Y_step + momentum * old_Y_step
                    Y_new = Y_new - new_Y_step
                    old_Y_step = new_Y_step
                    i += 1
            return Y_new
        else:
            return Y
    
    def set_projection(self, val='hardnetaff'):
        """set what projection to do during testing"""
        self._testProj = val
    
    apply_aff_projection = HardNetAff.apply_projection
    get_opt = HardNetCvx.get_opt
    apply_cvx_projection = HardNetCvx.apply_projection

    def forward(self, x, isTest=False):
        encoded_x = self._data.encode_input(x)
        out = self._net(encoded_x)
        # completion
        if self._args['useCompl']:
            out = self._data.complete_partial(x, out)
        # correction
        out = self.grad_steps(x, out)
        if isTest:
            out = self.extra_grad_steps(x, out)
        
        if isTest and self._testProj == 'hardnetaff':
            out = self.apply_aff_projection(out, x)
        elif isTest and self._testProj == 'hardnetcvx':
            out = self.apply_cvx_projection(out, x)
        
        return out

if __name__=='__main__':
    main()