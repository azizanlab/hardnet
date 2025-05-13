try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import pickle
from setproctitle import setproctitle
import os
import argparse
import cvxpy as cp
import numpy as np

from utils import add_common_args, get_dict_from_parser, load_data

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    name = 'baselineCBFQP'
    args = get_args(name)
    setproctitle(name+'-{}'.format(args['probType']))
    data = load_data(args, DEVICE)

    save_dir = os.path.join('results', str(data), name+args['suffix'], f"seed{args['seed']}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Save an empty net to fit into the common evaluation framework
    net = CBFQP(data, args)
    with open(os.path.join(save_dir, 'net.dict'), 'wb') as f:
        torch.save(net.state_dict(), f)
    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump({'train_time':np.array([0]), 'train_loss':np.array([0])}, f)


def get_args(name):
    parser = argparse.ArgumentParser(description=name)
    parser = add_common_args(parser)
    args = get_dict_from_parser(parser, name)
    print(f'{name}: {args}')
    return args


######### Models

class CBFQP(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
    
    def solve_single(self, x1):
        """solve one QP / x1: torch tensor, shape (1, 5)"""
        # build constraint pieces in torch
        A, _, bu = self._data.get_coefficients(x1)        # (1, m, ydim)
        u_nom     = self._data.get_nominal_control(x1)    # (1, ydim)

        # hand NumPy copies to CVXPY
        A_np   = A.squeeze(0).cpu().numpy()
        b_np   = bu.squeeze(0).cpu().numpy()
        u_nom  = u_nom.squeeze(0).cpu().numpy()

        # solve the QP
        u = cp.Variable(self._data.ydim)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_nom)),
                          [A_np @ u <= b_np])
        prob.solve(solver="OSQP", eps_abs=1e-6, eps_rel=1e-6, verbose=False)

        # return as tensor in the original dtype/device
        return torch.as_tensor(u.value, dtype=x1.dtype, device=x1.device)
    
    def forward(self, x, isTest=True):
        """forward over a batch / x: (B, 5) tensor"""
        outs = [self.solve_single(x[i:i+1])   # keep batch dim with i:i+1
                for i in range(len(x))]
        return torch.stack(outs, dim=0)

if __name__=='__main__':
    main()