try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.optim as optim
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import sys
import pickle
import pandas as pd
import argparse

from utils import load_data, eval_net
import hardnet_aff
import hardnet_cvx
import baseline_dc3
import baseline_nn
import baseline_cbfqp

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='testNets')
    parser.add_argument('--probType', type=str, default='pwc',
        choices=['pwc', 'pwcfull', 'pwcbox', 'opt', 'cbf'], help='problem type')
    parser.add_argument('--expDir', type=str,
        help='path for experimental results ex)results/NonconvexOpt-10000')
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    test_results(args['expDir'], args['probType'])


def load_net_hardnetaff(data, args, path):
    net = hardnet_aff.HardNetAff(data, args).to(DEVICE)
    with open(os.path.join(path), 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=DEVICE))
    net.set_projection(True)
    return net

def load_net_hardnetcvx(data, args, path):
    net = hardnet_cvx.HardNetCvx(data, args).to(DEVICE)
    with open(os.path.join(path), 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=DEVICE))
    net.set_projection(True)
    return net

def load_net_dc3(data, args, path):
    net = baseline_dc3.DC3(data, args).to(DEVICE)
    with open(os.path.join(path), 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=DEVICE))
    return net

def load_net_nn(data, args, path):
    net = baseline_nn.NN(data, args).to(DEVICE)
    with open(os.path.join(path), 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=DEVICE))
    return net

def load_net_cbfqp(data, args, path):
    net = baseline_cbfqp.CBFQP(data, args).to(DEVICE)
    with open(os.path.join(path), 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=DEVICE))
    return net


def test_results(exp_dir, prob_type):
    """
    Get dictionaries with experiment status and summary stats for all methods
    exp_dir: directory for entire experiment
    prob_type: problem type
    """
    torch.manual_seed(123)
    np.random.seed(123)

    ## load data through hardnet_aff's interface
    sys.argv = ['hardnet_aff.py', '--probType', prob_type]
    args = hardnet_aff.get_args('hardnetAff')
    data = load_data(args, DEVICE)
    test_dataset = TensorDataset(data.testX, data.testY)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    if os.path.exists(exp_dir):
        ## Get mapping of subdirs to methods
        methods = os.listdir(exp_dir)
        for method in methods:
            if 'Opt' in method:
                print(f'Skipped {method}.')
                continue
            print(f'Doing {method}...')

            path_method = os.path.join(exp_dir, method)
            instances = os.listdir(path_method)
            for inst in instances:
                result_dir = os.path.join(path_method, inst)
                file_path = os.path.join(result_dir, 'net.dict')
                if os.path.exists(file_path):
                    if 'hardnetAff' in method:
                        sys.argv = ['hardnet_aff.py', '--probType', prob_type]
                        args = hardnet_aff.get_args('hardnetAff')
                        net = load_net_hardnetaff(data, args, file_path)
                    elif 'hardnetCvx' in method:
                        sys.argv = ['hardnet_cvx.py', '--probType', prob_type]
                        args = hardnet_cvx.get_args('hardnetCvx')
                        net = load_net_hardnetcvx(data, args, file_path)
                    elif 'DC3' in method:
                        if 'testProj' in method:
                            sys.argv = ['baseline_dc3.py', '--probType', prob_type,'--useTestProj', 'True']
                        else:
                            sys.argv = ['baseline_dc3.py', '--probType', prob_type]
                        args = baseline_dc3.get_args('baselineDC3')
                        net = load_net_dc3(data, args, file_path)
                    elif 'NN' in method:
                        if 'testProj' in method:
                            sys.argv = ['baseline_nn.py', '--probType', prob_type,'--useTestProj', 'True']
                        else:
                            sys.argv = ['baseline_nn.py', '--probType', prob_type]
                        args = baseline_nn.get_args('baselineNN')
                        net = load_net_nn(data, args, file_path)
                    elif 'baselineCBFQP' in method:
                        sys.argv = ['baseline_cbfqp.py', '--probType', prob_type]
                        args = baseline_cbfqp.get_args('baselineCBFQP')
                        net = load_net_cbfqp(data, args, file_path)
                    else:
                        print(f'Testing for {method} is not supported.')
                        continue
                    
                    # Eval over test set
                    stats = {}
                    net.eval()
                    for Xtest, Ytarget_test in test_loader:
                        Xtest = Xtest.to(DEVICE)
                        Ytarget_test = Ytarget_test.to(DEVICE)
                        eval_net(data, Xtest, Ytarget_test, net, args, 'test', stats)

                    for key in stats.keys():
                        # match dimensions of stats to that of training stats (axis 0 is for epoch)
                        stats[key] = np.expand_dims(np.array(stats[key]), axis=0)
                    
                    with open(os.path.join(result_dir, 'test_stats.dict'), 'wb') as f:
                        pickle.dump(stats, f)
                    
                    del net
                    torch.cuda.empty_cache()
                else:
                    print(f'Not Found: {file_path}')

if __name__=='__main__':
    main()