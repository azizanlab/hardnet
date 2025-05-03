import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from toy_problem import ToyProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

def generate_toy_dataset(num_examples):
    print(f"neq:1, eq:0")
    np.random.seed(17)
    X = np.random.uniform(-1.2, 1.2, size=(num_examples, 1))

    problem = ToyProblem(X)
    print(f"problem length:{len(problem.Y)}")

    with open(f"./toy_dataset_var1_ineq1_eq0_ex{num_examples}", 'wb') as f:
        pickle.dump(problem, f)


num_examples = 50
generate_toy_dataset(num_examples)