import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from pwcbox_problem import PWCBoxProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

def generate_toy_dataset(num_examples):
    np.random.seed(17)
    X = np.random.uniform(-2, 2, size=(num_examples, 1))

    problem = PWCBoxProblem(X)
    print(f"problem length:{len(problem.Y)}")

    with open(f"./pwcbox_dataset_ex{num_examples}", 'wb') as f:
        pickle.dump(problem, f)


num_examples = 10
generate_toy_dataset(num_examples)