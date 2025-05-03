import torch
torch.set_default_dtype(torch.float64)

import numpy as np
# import ipopt
from gekko import GEKKO
import time

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###################################################################
# LEARNING NONCONVEX OPTIMIZATION SOLVER
###################################################################

class NonconvexProblem:
    """
        Learning a single solver (input:x, output: solution)
        for the following nonconvex problems for all x:
        minimize_y 1/2 * y^T Q y + p^T sin(y)
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
        return f'NonconvexOpt-{self.ydim}-{self.nineq}-{self.neq}-{self.num}'

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
        return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)

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
        return A_eff.repeat(X.size()[0],1,1), b_eff

    def get_ineq_partial_grad(self, X, Y):
        A_eff, b_eff = self.get_Ab_effective(X)
        A_eff = A_eff[0,:,:]
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

    def opt_solve(self, X, solver_type='gekko', tol=1e-4):
        Q, p, A, b, C = self.Q_np, self.p_np, self.A_np, self.b_np, self.C_np
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        for Xi in X_np:
            if solver_type == 'ipopt':
                y0 = np.linalg.pinv(C)@Xi  # feasible initial point

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([Xi, -np.inf * np.ones(A.shape[0])])
                cu = np.hstack([Xi, b])

                nlp = ipopt.problem(
                            n=len(y0),
                            m=len(cl),
                            problem_obj=nonconvex_ipopt(Q, p, C, A),
                            lb=lb,
                            ub=ub,
                            cl=cl,
                            cu=cu
                            )

                nlp.addOption('tol', tol)
                nlp.addOption('print_level', 0) # 3)

                start_time = time.time()
                y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                total_time += (end_time - start_time)
            elif solver_type == 'gekko':
                y0 = np.linalg.pinv(C)@Xi  # feasible initial point

                m = GEKKO(remote=False)
                m.options.OTOL = tol
                m.options.RTOL = tol

                y = m.Array(m.Var, y0.shape, lb=-1e5, ub=1e5)

                for j in range(len(y0)):
                    y[j].value = y0[j]

                for j in range(len(A)):
                    m.Equation(np.dot(A[j,:], y) <= b[j])
                
                for j in range(len(C)):
                    m.Equation(np.dot(C[j,:], y) == Xi[j])
                
                nonconvex_term = sum(p[i] * m.sin(y[i]) for i in range(len(y)))

                m.Minimize(m.sum(0.5 * (y * np.matmul(Q, y))) + nonconvex_term)

                start_time = time.time()
                m.solve(disp=False)
                end_time = time.time()
                Y.append([yi.value[0] for yi in y])
                total_time += (end_time - start_time)
            else:
                raise NotImplementedError

        return np.array(Y), total_time, total_time/len(X_np)

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class nonconvex_ipopt(object):
    def __init__(self, Q, p, C, A):
        self.Q = Q
        self.p = p
        self.C = C
        self.A = A
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@np.sin(y)

    def gradient(self, y):
        return self.Q@y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.C@y, self.A@y])

    def jacobian(self, y):
        return np.concatenate([self.C.flatten(), self.A.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    # def intermediate(self, alg_mod, iter_count, obj_value,
    #         inf_pr, inf_du, mu, d_norm, regularization_size,
    #         alpha_du, alpha_pr, ls_trials):
    #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))