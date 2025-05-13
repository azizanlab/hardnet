import torch
import torch.optim as optim
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import time
import os
import sys
import pickle
import pandas as pd

import default_args

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def add_common_args(parser):
    """add common arguments to parser"""
    parser.add_argument('--probType', type=str, default='pwc',
        choices=['pwc', 'pwcfull', 'pwcbox', 'opt', 'cbf'], help='problem type')
    parser.add_argument('--suffix', type=str, default='',
        help='suffix for method name (start with _)')
    parser.add_argument('--seed', type=int, default=123,
        help='random seed for reproducibility')
    parser.add_argument('--nEx', type=int,
        help='total number of datapoints')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--saveAllStats', type=bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int, default=100,
        help='how frequently (in terms of number of epochs) to save stats to file')
    parser.add_argument('--evalFreq', type=int, default=1,
        help='how frequently (in terms of number of epochs) to evaluate the learned model over the valid/test set')
    parser.add_argument('--softWeight', type=float,
        help='weight given to the regularization term in loss for penalizing constraint violations')
    return parser

def get_dict_from_parser(parser, method_name):
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = getattr(default_args, method_name+'_default_args')(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    return args

def load_data(args, device):
    """Load data, and put on GPU if needed"""
    prob_type = args['probType']
    filepath = os.path.join(
        'datasets', prob_type, f"{prob_type}_dataset_ex{args['nEx']}"
    )
    with open(filepath, 'rb') as f:
        sys.path.append('./datasets/'+prob_type)
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(device))
            except AttributeError:
                pass
    data._device = device
    
    return data

def agg_dict(stats, key, value, op='concat'):
    """Modifies stats in place"""
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

def record_stats(stats, runtime, eval_metric, err1, err2, prefix):
    make_prefix = lambda x: f"{prefix}_{x}"
    agg_dict(stats, make_prefix('time'), runtime, op='sum')
    agg_dict(stats, make_prefix('eval'), eval_metric)
    agg_dict(stats, make_prefix('err1_max'), np.max(err1, axis=1))
    agg_dict(stats, make_prefix('err1_mean'), np.mean(err1, axis=1))
    agg_dict(stats, make_prefix('err1_nviol'), np.sum(err1 > 1e-4, axis=1))
    agg_dict(stats, make_prefix('err2_max'), np.max(err2, axis=1))
    agg_dict(stats, make_prefix('err2_mean'), np.mean(err2, axis=1))
    agg_dict(stats, make_prefix('err2_nviol'), np.sum(err2 > 1e-4, axis=1))
    return stats

def eval_net(data, X, Ytarget, net, args, prefix, stats):
    with torch.no_grad():
        _ = net(X, isTest=True) # warm up once so autotuners do not pollute timing

        if torch.cuda.is_available():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize() # clear any previous work
            starter.record()
            Y = net(X, isTest=True)
            ender.record()
            torch.cuda.synchronize()
            runtime = starter.elapsed_time(ender) / 1000.0 # convert to seconds
        else:
            start_time = time.time()
            Y = net(X, isTest=True)
            runtime = time.time() - start_time

    eval_metric = data.get_eval_metric(net, X, Y, Ytarget).detach().cpu().numpy()
    err1 = data.get_err_metric1(net, X, Y, Ytarget).detach().cpu().numpy()
    err2 = data.get_err_metric2(net, X, Y, Ytarget).detach().cpu().numpy()

    return record_stats(stats, runtime, eval_metric, err1, err2, prefix)

def train_net(data, args, net_cls, save_dir, net_modifier_fn=None):
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']

    # for reproducibility
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    train_dataset = TensorDataset(data.trainX, data.trainY)
    valid_dataset = TensorDataset(data.validX, data.validY)
    test_dataset = TensorDataset(data.testX, data.testY)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    net = net_cls(data, args)
    net.to(DEVICE)
    solver_opt = optim.Adam(net.parameters(), lr=solver_step)

    stats = {}
    for epoch in range(nepochs):
        epoch_stats = {}

        if net_modifier_fn is not None:
            net = net_modifier_fn(net, epoch)

        # Train
        net.train()
        for Xtrain, Ytarget_train in train_loader:
            Xtrain = Xtrain.to(DEVICE)
            Ytarget_train = Ytarget_train.to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            train_loss = data.get_train_loss(net, Xtrain, Ytarget_train, args)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            agg_dict(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            agg_dict(epoch_stats, 'train_time', train_time, op='sum')
        
        # Print results
        print(f"Epoch {epoch}: train loss {np.mean(epoch_stats['train_loss']):.2f},"\
              f"train time {np.mean(epoch_stats['train_time']):.3f}")
        
        if epoch % args['evalFreq'] == 0:
            # Eval over valid set
            net.eval()
            for Xvalid, Ytarget_valid in valid_loader:
                Xvalid = Xvalid.to(DEVICE)
                Ytarget_valid = Ytarget_valid.to(DEVICE)
                eval_net(data, Xvalid, Ytarget_valid, net, args, 'valid', epoch_stats)

            # # Eval over test set
            # net.eval()
            # for Xtest, Ytarget_test in test_loader:
            #     Xtest = Xtest.to(DEVICE)
            #     Ytarget_test = Ytarget_test.to(DEVICE)
            #     eval_net(data, Xtest, Ytarget_test, net, args, 'test', epoch_stats)

            print(f"eval {np.mean(epoch_stats['valid_eval']):.2f}, "\
                f"err1 max {np.mean(epoch_stats['valid_err1_max']):.2f}, "\
                f"err1 mean {np.mean(epoch_stats['valid_err1_mean']):.2f}, "\
                f"err1 nviol {np.mean(epoch_stats['valid_err1_nviol']):.2f}, "\
                f"err2 max {np.mean(epoch_stats['valid_err2_max']):.2f}, "\
                f"err2 mean {np.mean(epoch_stats['valid_err2_mean']):.2f}, "\
                f"err2 nviol {np.mean(epoch_stats['valid_err2_nviol']):.2f}, "\
                f"time {np.mean(epoch_stats['valid_time']):.3f}"
            )

        if args['saveAllStats']:
            if epoch == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            # stats = epoch_stats
            for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)

        if (epoch % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, f'net_epoch{epoch}.dict'), 'wb') as f:
                torch.save(net.state_dict(), f)

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'net.dict'), 'wb') as f:
        torch.save(net.state_dict(), f)
    
    return net, stats

#############################
#### For loading results ####
#############################

def get_results(exp_dir, total_epoch=1000):
    """
    Get dictionaries with experiment status and summary stats for all methods
    If a method has multiple setups, choose first
    exp_dir: directory for entire experiment
    """
    exp_status_dict = {}
    stats_dict = {}
    
    if os.path.exists(exp_dir):
        ## Get mapping of subdirs to methods
        methods = os.listdir(exp_dir)
        for method in methods:
            print(f'Reading {method}...')
            path_method = os.path.join(exp_dir, method)
            aggregate_for_method(method, path_method, exp_status_dict, stats_dict, total_epoch)
        
    return exp_status_dict, stats_dict


def aggregate_for_method(method_name, path_method, exp_status_dict, stats_dict, total_epoch=1000):
    """
    Fill in passed in exper_status and stats dicts with status/summary stats info
    Results are aggregated across different instances (repeats)
    """
    method_stats = []
    instances = os.listdir(path_method)
    num_running, num_done = 0, 0
    
    # get status and stats
    for inst in instances:
        print(f'  Reading {inst}...')
        result_dir = os.path.join(path_method, inst)
        is_done, stats = check_running_done(result_dir, 'Opt' in method_name, total_epoch)
        if is_done:
            num_done += 1
            method_stats.append(stats)
        else:
            num_running += 1
    exp_status_dict[method_name] = (num_running, num_done)

    # aggregate metrics
    d = {}
    if len(method_stats) > 0:
        metrics = method_stats[0].keys()
        if 'Opt' not in method_name:
            for metric in metrics:
                d[metric] = get_mean_std_nets(method_stats, metric)
        else:
            for metric in metrics:
                d[metric] = get_mean_std_opts(method_stats, metric)
    stats_dict[method_name] = d
    


def check_running_done(path, is_opt=False, total_epoch=1000):
    """
    Check if experiment is running or done, and return stats if done
    is_opt: whether this method is optimizer baseline
    """
    is_done = False
    stats = None
    
    if is_opt:
        if os.path.exists(os.path.join(path, 'results.dict')):
            with open(os.path.join(path, 'results.dict'), 'rb') as f:
                stats = pickle.load(f)
            is_done = True
    else:        
        try:   
            if os.path.exists(os.path.join(path, 'stats.dict')):
                with open(os.path.join(path, 'stats.dict'), 'rb') as f:
                    stats = pickle.load(f)
                with open(os.path.join(path, 'test_stats.dict'), 'rb') as f:
                    test_stats = pickle.load(f)
                stats.update(test_stats)
                is_done = ('baselineCBFQP' in path or len(stats['valid_time']) >= total_epoch)
                if not is_done:
                    print(path)
                    print(f"Not enough epochs! Last epoch: {len(stats['valid_time'])}")
        except Exception as e:
            print(str(e))
            is_done = False
            stats  = None
            
    return is_done, stats


def get_mean_std_nets(stats_dicts, metric):
    """
    Compute summary stats for neural network methods
    Note: Assumes stats are saved for each epoch 
          (i.e., saveAllStats flag is True for each run)
    """
    if 'train_time' in metric:
        results = [d[metric].sum() for d in stats_dicts]
    elif 'time' in metric:
        # test and valid time: use time for latest epoch
        results = [d[metric][-1] for d in stats_dicts]
    else:
        # use mean across samples for latest epoch
        results = [d[metric][-1].mean() for d in stats_dicts]

    # return mean and stddev across replicates
    return np.mean(results), np.std(results)


def get_mean_std_opts(stats_dicts, metric):
    """
    Compute summary stats for baseline optimizers
    """
    if 'time' in metric:
        results = [d[metric] for d in stats_dicts]
    else:
        results = [d[metric].mean() for d in stats_dicts]
    return np.mean(results), np.std(results)


def get_table_from_dict(stats_dict, metrics, keep_methods, test_time_unit='s', train_time_unit='s'):
    """
    Make a table (data frame) from the result dictionary
    """
    d = {}
    missing_methods = []
    for method in keep_methods:
        if method in stats_dict and stats_dict[method] != {}:
            if 'Opt' in method:
                stats_dict[method]['train_time'] = 0, 0
            test_time_scale = 1000 if test_time_unit == 'ms' else 1/60 if test_time_unit == 'min' else 1
            train_time_scale = 1000 if train_time_unit == 'ms' else 1/60 if train_time_unit == 'min' else 1
            d[method] = ['{:.2f} ({:.2f})'.format(*[test_time_scale*t for t in stats_dict[method][metric]]) if 'test_time' in metric else \
                         '{:.2f} ({:.2f})'.format(*[train_time_scale*t for t in stats_dict[method][metric]]) if 'train_time' in metric else \
                         '{:.2f} ({:.2f})'.format(*stats_dict[method][metric]) for metric in metrics]
        else:
            missing_methods.append(method)
    df = pd.DataFrame.from_dict(d, orient='index')
    df.columns = metrics
    df.index.names = ['alg']
    if len(missing_methods) > 0:
        print(f'missing methods: {missing_methods}')
    return df.loc[[x for x in keep_methods if x not in missing_methods]], missing_methods


def get_latex_from_table(df):
    print(df.style.to_latex().replace('\$', '$').replace('\_', '_'))
