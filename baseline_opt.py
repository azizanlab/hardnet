try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import add_common_args, get_dict_from_parser, load_data, record_stats

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    name = 'baselineOpt'
    args = get_args(name)
    setproctitle(name+'-{}'.format(args['probType']))
    data = load_data(args, DEVICE)

    ## Run pure optimization baselines
    prob_type = args['probType']
    if prob_type == 'qp':
        solvers = ['osqp']
    elif prob_type == 'nonconvex':
        solvers = ['gekko']
    else:
        raise NotImplementedError

    for solver in solvers:
        save_dir = os.path.join('results', str(data), name+'_{}'.format(solver),
            time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        opt_results = {}
        # Yvalid_opt, valid_time_total, valid_time_parallel = data.opt_solve(data.validX, solver_type=solver, tol=args['tol'])
        # opt_results = get_opt_results(opt_results, data, data.validX, torch.tensor(Yvalid_opt).to(DEVICE),
        #                                 data.validY, valid_time_parallel, 'valid')
        # opt_results.update(dict([('valid_time_total', valid_time_total)]))
        Ytest_opt, test_time_total, test_time_parallel = data.opt_solve(data.testX, solver_type=solver, tol=args['tol'])
        opt_results = get_opt_results(opt_results, data, data.testX, torch.tensor(Ytest_opt).to(DEVICE),
                                        data.testY, test_time_parallel, 'test')
        opt_results.update(dict([('test_time_total', test_time_total)]))
        with open(os.path.join(save_dir, 'results.dict'), 'wb') as f:
            pickle.dump(opt_results, f)

def get_args(name):
    parser = argparse.ArgumentParser(description=name)
    parser = add_common_args(parser)
    parser.add_argument('--tol', type=float,
        help='tolerance for optimization')
    args = get_dict_from_parser(parser, name)
    print(f'{name}: {args}')
    return args

def get_opt_results(results, data, X, Y, Ytarget, time, prefix):
    eval_metric = data.get_eval_metric(None, X, Y, Ytarget).detach().cpu().numpy()
    ineq_err = data.get_ineq_error(None, X, Y, Ytarget).detach().cpu().numpy()
    eq_err = data.get_eq_error(None, X, Y, Ytarget).detach().cpu().numpy()
    results = record_stats(results, time, eval_metric, ineq_err, eq_err, prefix)

    return results

if __name__=='__main__':
    main()