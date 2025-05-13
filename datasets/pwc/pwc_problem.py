import torch
torch.set_default_dtype(torch.float64)

import numpy as np

import time
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###################################################################
# Learning with Piecewise Constraints
###################################################################

class PWCProblem:
    def __init__(self, X, valid_frac=0.0, test_frac=0.0):
        self._X = torch.tensor(X)
        self._Y = self.calc_Y(self._X)
        self._encoded_xdim = X.shape[1]
        self._ydim = self._Y.shape[1]
        self._num = X.shape[0]
        self._neq = 0
        self._partial_vars = np.arange(self._ydim)
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._X_eval = torch.linspace(-2,2,401)[:,None]
        self._Y_eval = self.calc_Y(self._X_eval)

        ### For Pytorch
        self._device = None

    def __str__(self):
        return f'PWCProblem-{self.num}'
    
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
        area3 = (X>0)*(X<=1)*(4-9*(X-2/3)**2)
        area4 = (X>1)*(8-5*X)
        return area1 + area2 + area3 + area4
    
    def get_lower_bound(self, X):
        """lower bound for inequality constraints bl(x)<=A(x)y<=bu(x)"""
        largeNum = 1e10
        return torch.zeros(X.shape[0], 1, device=self.device) -largeNum
    
    def get_upper_bound(self, X):
        """upper bound for inequality constraints bl(x)<=A(x)y<=bu(x)"""
        area1 = (X<=-1)*(-5*torch.sin(torch.pi*(X+1)/2)**2)
        area2 = (X>-1)*(X<=0)*(0)
        area3 = (X>0)*(X<=1)*(9*(X-2/3)**2-4)*X
        area4 = (X>1)*(7.5-4.5*X)
        return area1 + area2 + area3 + area4

    def get_boundaries(self, X):
        """for visualization"""
        area1 = (X<=-1)*(5*torch.sin(torch.pi*(X+1)/2)**2)
        area2 = (X>-1)*(X<=0)*(0)
        area3 = (X>0)*(X<=1)*(4-9*(X-2/3)**2)*X
        area4 = (X>1)*(7.5-4.5*X)
        return area1 + area2 + area3 + area4

    def get_resid(self, X, Y):
        A, bl, bu = self.get_coefficients(X)
        resid_upper = torch.clamp((A@Y[:,:,None])[:,:,0] - bu, 0)
        return resid_upper # resid_lower is always 0
    
    def get_train_loss(self, net, X, Ytarget, args):
        Y = net(X)
        main_loss = torch.norm(Ytarget - Y, dim=1)**2
        regularization = torch.norm(self.get_resid(X, Y), dim=1)**2
        return main_loss + args['softWeight'] * regularization
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return (Ytarget - Y)**2
    
    def get_err_metric1(self, net, X, Y, Ytarget):
        """compute ineq error over eval samples"""
        Y_new = net(self._X_eval, isTest=True)
        resid = self.get_resid(self._X_eval, Y_new).squeeze()
        return resid.repeat(X.shape[0], 1)
    
    def get_err_metric2(self, net, X, Y, Ytarget):
        return torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

    def get_coefficients(self, X):
        """coefficients for inequality constraints bl(x)<=A(x)y<=bu(x)"""
        A = (X<=-1)*(-1.0) + (X>-1)*(X<=0)*(1.0) + (X>0)*(X<=1)*(-1.0) + (X>1)*1.0
        A = A[:,None,:]
        bl = self.get_lower_bound(X)
        bu = self.get_upper_bound(X)
        return A, bl, bu


    ##### For DC3 #####

    def get_resid_grad(self, X, Y):
        """gradient of ||resid||^2 = ||min(bl - A*y, 0)||^2 + ||min(A*y - bu, 0)||^2"""
        A, bl, bu = self.get_coefficients(X)
        resid_lower = torch.clamp(bl - (A@Y[:,:,None])[:,:,0], 0)
        resid_upper = torch.clamp((A@Y[:,:,None])[:,:,0] - bu, 0)
        return 2*(resid_lower[:,None,:]@A)[:,0,:] + 2*(resid_upper[:,None,:]@A)[:,0,:]

    def get_ineq_partial_grad(self, X, Y):
        return NotImplementedError

    def complete_partial(self, X, Z):
        return NotImplementedError