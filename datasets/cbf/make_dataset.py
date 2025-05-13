import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from safe_control import Obstacle, Unicycle_Acc, SafeControl

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

def generate_cbf_dataset(num_examples):
    np.random.seed(17)
    loss_max, alpha, kappa, T, dt = 20000, 20, 10, 1.0, 0.02 # alpha 10 kappa 5

    Q = np.diag([100, 100, 0.0, 0.1, 0.1])
    R = np.diag([0.1, 0.1])
    init_box = [np.array([-4.0, 0.0, -np.pi/4, 0.0, 0.0]),
                np.array([-3.5, 0.5, -np.pi/8, 0.0, 0.0])]
    sys = Unicycle_Acc(init_box, kappa=kappa)

    obs_param_list = [(-2.25, 0.5, 0.5, 0.75), (-0.8, 0., 0.5, 0.5)]
    obs_list = []
    num_obs = len(obs_param_list)
    for i in range(num_obs):
        obs_list.append(Obstacle(*obs_param_list[i]))

    X = sys.generate_states(num_examples)
    failed_indices = np.array(range(num_examples))
    while len(failed_indices) > 0:
        failed_samples = X[failed_indices, :]
        failure = np.repeat(False, len(failed_indices))
        for obs in obs_list:
            cbf = sys.get_cbf_h(torch.tensor(failed_samples), obs)
            failure = failure | (cbf.numpy() < 0)
        failed_indices = failed_indices[failure]
        X[failed_indices, :] = sys.generate_states(len(failed_indices))
        print(f'regenerated {len(failed_indices)} samples')

    problem = SafeControl(Q, R, X, sys, obs_list, loss_max, alpha, T, dt)
    print(f"problem length:{num_examples} examples with {int(T/dt)} steps")

    with open(f"./cbf_dataset_ex{num_examples}", 'wb') as f:
        pickle.dump(problem, f)

num_examples = 1000

generate_cbf_dataset(num_examples)
