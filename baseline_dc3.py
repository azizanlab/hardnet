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
    name = 'baselineDC3'
    args = get_args(name)
    setproctitle(name+'-{}'.format(args['probType']))
    data = load_data(args, DEVICE)

    save_dir = os.path.join('results', str(data), name+args['suffix'],
        time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Run method
    train_net(data, args, NNSolver, save_dir)


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
    parser.add_argument('--useTestProj', type=bool,
        help='whether to use projection during testing')
    args = get_dict_from_parser(parser, name)
    print(f'{name}: {args}')
    return args


######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        self._if_project = args['useTestProj']
        layer_sizes = [data.encoded_xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        output_dim = data.ydim - data.nknowns
        if self._args['useCompl']:
            output_dim -= data.neq
        
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

        self.net = nn.Sequential(*layers)

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
                    ineq_step = self._data.get_ineq_grad(X, Y_new)
                    eq_step = self._data.get_eq_grad(X, Y_new)
                    Y_step = (1 - self._args['softWeightEqFrac']) * ineq_step + self._args['softWeightEqFrac'] * eq_step
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
                while (i == 0 or torch.max(self._data.get_eq_resid(X, Y_new)) > eps_converge or
                        torch.max(self._data.get_ineq_resid(X, Y_new)) > eps_converge) and i < max_steps:
                    if partial_corr:
                        Y_step = self._data.get_ineq_partial_grad(X, Y_new)
                    else:
                        ineq_step = self._data.get_ineq_grad(X, Y_new)
                        eq_step = self._data.get_eq_grad(X, Y_new)
                        Y_step = (1 - self._args['softWeightEqFrac']) * ineq_step + self._args['softWeightEqFrac'] * eq_step
                    new_Y_step = lr * Y_step + momentum * old_Y_step
                    Y_new = Y_new - new_Y_step
                    old_Y_step = new_Y_step
                    i += 1
            return Y_new
        else:
            return Y
    
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
        out = self.net(encoded_x)
        # completion
        if self._args['useCompl']:
            out = self._data.complete_partial(x, out)
        else:
            out = self._data.process_output(x, out)
        # correction
        out = self.grad_steps(x, out)
        if isTest:
            out = self.extra_grad_steps(x, out)
        
        if self._if_project:
            out = out[:, self._data.partial_vars]
            A_eff, b_eff = self._data.get_Ab_effective(x)
            out = self.apply_projection(out, A_eff, b_eff)
            out = self._data.complete_partial(x, out)

        return out

if __name__=='__main__':
    main()