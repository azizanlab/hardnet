import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from nonconvex_problem import NonconvexProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 10000

np.random.seed(17)

Q = np.diag(np.random.random(num_var))
p = np.random.random(num_var)
A = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
C = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
b = np.sum(np.abs(A@np.linalg.pinv(C)), axis=1)
X = np.random.uniform(-1, 1, size=(num_examples, num_eq))

problem = NonconvexProblem(Q, p, A, b, C, X)
print(len(problem.Y))

with open(f"./opt_dataset_ex{num_examples}", 'wb') as f:
    pickle.dump(problem, f)
