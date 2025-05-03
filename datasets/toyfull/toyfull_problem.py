import torch
torch.set_default_dtype(torch.float64)

import numpy as np

import time
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###################################################################
# Learning with Piecewise Constraints
###################################################################

class ToyFullProblem:
    """ 
        y = 1 - cos(kx) + alpha x  if x>=0
            cos(kx) - 1 + beta x  if x<0
        x in [-2, 2]
    """
    def __init__(self, X, valid_frac=0.0, test_frac=0.0):
        self._X = torch.tensor(X)
        self._Y = self.calc_Y(self._X)
        self._encoded_xdim = X.shape[1]
        self._ydim = self._Y.shape[1]
        self._num = X.shape[0]
        self._nineq = 1
        self._neq = 0
        self._nknowns = 0
        self._partial_vars = np.arange(self._ydim)
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._X_eval = torch.linspace(-2,2,401)[:,None]
        self._Y_eval = self.calc_Y(self._X_eval)
        self._dx = 0.01

        ### For Pytorch
        self._device = None

    def __str__(self):
        return f'ToyFullProblem-{self.ydim}-{self.nineq}-{self.neq}-{self.num}'
    
    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y
    
    @property
    def X_eval(self):
        return self._X_eval

    @property
    def Y_eval(self):
        return self._Y_eval
    
    @property
    def partial_unknown_vars(self):
        return self._partial_vars

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
        return self._X_eval

    @property
    def testX(self):
        return self._X_eval

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self._Y_eval

    @property
    def testY(self):
        return self._Y_eval

    @property
    def device(self):
        return self._device
    
    def encode_input(self, X):
        return X
    
    def calc_Y(self, X):
        area1 = (X<=-1)*(-5*torch.sin(torch.pi*(X+1)/2))
        area2 = (X>-1)*(X<=0)*(0)
        area3 = (X>0)*(X<=1)*(-9*(X-2/3)**2+4)
        area4 = (X>1)*(-5*(X-1)+3)
        return area1 + area2 + area3 + area4
    
    def get_boundaries(self, X):
        area1 = (X<=-1)*(5*torch.sin(torch.pi*(X+1)/2)**2)
        area2 = (X>-1)*(X<=0)*(0)
        area3 = (X>0)*(X<=1)*(-9*(X-2/3)**2+4)*X
        area4 = (X>1)*(-4.5*(X-1)+3)
        return area1 + area2 + area3 + area4

    def get_ineq_resid(self, X, Y):
        A, b = self.get_Ab_effective(X)
        return torch.clamp((A@Y[:,:,None])[:,:,0] - b, 0)

    def get_eq_resid(self, X, Y):
        return torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

    def get_ineq_grad(self, X, Y):
        """gradient of ||ineq_resid||^2"""
        A, b = self.get_Ab_effective(X)
        ineq_resid = self.get_ineq_resid(X, Y)
        return 2*(ineq_resid[:,None,:]@A)[:,0,:]

    def get_eq_grad(self, X, Y):
        """gradient of ||eq_resid||^2"""
        return torch.zeros(X.shape[0], self._ydim, dtype=X.dtype, device=X.device)
    
    def get_train_loss(self, net, X, Ytarget, args):
        Y = net(X)
        main_loss = torch.norm(Ytarget - Y, dim=1)**2
        ineq_loss = torch.norm(self.get_ineq_resid(X, Y), dim=1)**2
        eq_loss = torch.norm(self.get_eq_resid(X, Y), dim=1)**2
        return main_loss + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_loss + \
            args['softWeight'] * args['softWeightEqFrac'] * eq_loss
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return (Ytarget - Y)**2
    
    def get_ineq_error(self, net, X, Y, Ytarget):
        """compute ineq error by treating each eval sample as a constraint"""
        Y_new = net(self._X_eval, isTest=True)
        resid = self.get_ineq_resid(self._X_eval, Y_new).squeeze()
        return resid.repeat(X.shape[0], 1)
    
    def get_eq_error(self, net, X, Y, Ytarget):
        return self.get_eq_resid(X, Y)

    def get_Ab_effective(self, X):
        """coefficients for reduced inequality constraint A_eff(x)y<=b_eff(x)"""
        A_eff = (X<=-1)*(-1.0) + (X>-1)*(X<=0)*(1.0) + (X>0)*(X<=1)*(-1.0) + (X>1)*1.0
        A_eff = A_eff[:,None,:]
        b_eff = (X<=-1)*(-5*torch.sin(torch.pi*(X+1)/2)**2) + (X>-1)*(X<=0)*(0) + (X>0)*(X<=1)*(9*(X-2/3)**2-4)*X + (X>1)*(-4.5*(X-1)+3)
        return A_eff, b_eff

    def get_ineq_partial_grad(self, X, Y):
        return NotImplementedError

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        return Z