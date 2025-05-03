import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
from scipy.sparse import csc_matrix

import time
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###################################################################
# Quadratic Program
###################################################################

class QuadraticProgram:
    """
        Learning a single solver (input:x, output: solution)
        for the following quadratic programs for all x:
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay <= b
                   Cy =  x
    """
    def __init__(self, Q, p, A, b, C, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._b = torch.tensor(b)
        self._C = torch.tensor(C)
        self._X = torch.tensor(X)
        self._Y = None
        self._encoded_xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._nineq = A.shape[0]
        self._neq = C.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._C[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._C_partial = self._C[:, self._partial_vars]
            self._C_other_inv = torch.inverse(self._C[:, self._other_vars])

        ### For Pytorch
        self._device = None

    def __str__(self):
        return f'QuadraticProgram-{self.ydim}-{self.nineq}-{self.neq}-{self.num}'

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def C(self):
        return self._C

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def b_np(self):
        return self.b.detach().cpu().numpy()

    @property
    def C_np(self):
        return self.C.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def encoded_xdim(self):
        return self._encoded_xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device
    
    def encode_input(self, X):
        return X

    def evaluate(self, X, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def get_ineq_resid(self, X, Y):
        return torch.clamp(Y@self.A.T - self.b, 0)

    def get_eq_resid(self, X, Y):
        return torch.abs(Y@self.C.T - X)

    def get_ineq_grad(self, X, Y):
        """gradient of ineq_resid^2"""
        ineq_resid = self.get_ineq_resid(X, Y)
        return 2*ineq_resid@self.A

    def eq_grad(self, X, Y):
        """gradient of eq_resid^2"""
        eq_resid = self.get_eq_resid(X, Y)
        return 2*(Y@self.C.T - X)@self.C
    
    def get_train_loss(self, net, X, Ytarget, args, coeff=1.0):
        Y = net(X)
        main_loss = self.evaluate(X, Y)
        ineq_loss = torch.norm(self.get_ineq_resid(X, Y), dim=1)**2
        eq_loss = torch.norm(self.get_eq_resid(X, Y), dim=1)**2
        return main_loss + coeff * args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_loss + \
            coeff * args['softWeight'] * args['softWeightEqFrac'] * eq_loss
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return self.evaluate(X, Y)
    
    def get_ineq_error(self, net, X, Y, Ytarget):
        return self.get_ineq_resid(X, Y)
    
    def get_eq_error(self, net, X, Y, Ytarget):
        return self.get_eq_resid(X, Y)

    def get_Ab_effective(self, X):
        """coefficients for reduced inequality constraint A_eff(x)y<=b_eff(x)"""
        A_eff = self.A[:, self.partial_vars] - self.A[:, self.other_vars] @ (self._C_other_inv @ self._C_partial)
        b_eff = self.b - (X @ self._C_other_inv.T) @ self.A[:, self.other_vars].T
        return A_eff, b_eff

    def get_ineq_partial_grad(self, X, Y):
        A_eff, b_eff = self.get_Ab_effective(X)
        grad_partial = 2 * torch.clamp(Y[:, self.partial_vars] @ A_eff.T - b_eff, 0) @ A_eff
        grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        grad[:, self.partial_vars] = grad_partial
        grad[:, self.other_vars] = - (grad_partial @ self._C_partial.T) @ self._C_other_inv.T
        return grad

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._C_partial.T) @ self._C_other_inv.T
        return Y

    def opt_solve(self, X, solver_type='osqp', tol=1e-4):

        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.A, self.b, self.C, X)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, b, C = \
                self.Q_np, self.p_np, self.A_np, self.b_np, self.C_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([C, A])
                my_l = np.hstack([Xi, -np.ones(b.shape[0]) * np.inf])
                my_u = np.hstack([Xi, b])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y