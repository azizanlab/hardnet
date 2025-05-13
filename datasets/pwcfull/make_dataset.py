import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from pwcfull_problem import PWCFullProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

def generate_dataset(num_examples):
    np.random.seed(17)
    X = np.random.uniform(-2.0, 2.0, size=(num_examples, 1))

    problem = PWCFullProblem(X)
    print(f"problem length:{len(problem.Y)}")

    with open(f"./pwcfull_dataset_ex{num_examples}", 'wb') as f:
        pickle.dump(problem, f)


num_examples = 50
generate_dataset(num_examples)