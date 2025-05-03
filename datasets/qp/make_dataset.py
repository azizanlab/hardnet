import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from quadratic_program import QuadraticProgram

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

def generate_qp_dataset(num_var, num_ineq, num_eq, num_examples):
    print(f"neq:{num_ineq}, eq:{num_eq}")
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    C = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    b = np.sum(np.abs(A@np.linalg.pinv(C)), axis=1)
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))

    problem = QuadraticProgram(Q, p, A, b, C, X)
    problem.calc_Y()
    print(f"problem length:{len(problem.Y)}")

    with open(f"./qp_dataset_var{num_var}_ineq{num_ineq}_eq{num_eq}_ex{num_examples}", 'wb') as f:
        pickle.dump(problem, f)

num_var = 100
num_examples = 10000

num_ineq = 50
for num_eq in [50]: #[10, 30, 50, 70, 90]:
    generate_qp_dataset(num_var, num_ineq, num_eq, num_examples)

# num_eq = 50
# for num_ineq in [10, 30, 70, 90]:
#     generate_qp_dataset(num_var, num_ineq, num_eq, num_examples)
