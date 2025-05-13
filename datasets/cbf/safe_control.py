import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import time
import os

from abc import ABC, abstractmethod

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

###################################################################
# Safe Control
###################################################################

class Obstacle:
    """elliptical obstacle with center at (cx,cy) and radius (rx, ry)"""
    def __init__(self, cx, cy, rx, ry):
        self._cx = cx
        self._cy = cy
        self._rx = rx
        self._ry = ry
    
    @property
    def cx(self):
        return self._cx
    
    @property
    def cy(self):
        return self._cy

    @property
    def rx(self):
        return self._rx

    @property
    def ry(self):
        return self._ry

class ControlAffineSystem(ABC):
    """system following x_dot = f(x) + g(x) u"""
    @property
    def encoded_state_dim(self):
        return self._encoded_state_dim

    @property
    def control_dim(self):
        return self._control_dim
    
    @abstractmethod
    def get_f(self, X):
        """output size: batch_size x state_dim"""
        pass
    
    @abstractmethod
    def get_g(self, X):
        """output size: batch_size x state_dim x control_dim"""
        pass

    @abstractmethod
    def get_cbf_h(self, X, obs):
        pass

    @abstractmethod
    def get_cbf_h_grad(self, X, obs):
        pass

    @abstractmethod
    def get_nominal_control(self, X):
        pass

    def encode_input(self, X):
        return X

    def get_control(self, net, X, isTest):
        return net(X, isTest=isTest)

    def get_xdot(self, X, U):
        return self.get_f(X) + (self.get_g(X) @ U[:,:,None])[:,:,0]

    def generate_states(self, nStates):
        prop = np.random.rand(nStates, self._state_dim)
        states = prop * self._init_box[0] + (1-prop) * self._init_box[1]
        return states

    def step(self, X, dt, control_fn, cost_fn, step_type='adaptive'):
        """control_fn: addition to nominal control"""
        if step_type == 'euler':
            output = control_fn(X)
            U = self.get_nominal_control(X) + output
            X_next = X + dt * self.get_xdot(X, U)
            cost = dt * cost_fn(X, output)
        
        elif step_type == 'adaptive':
            X_next = X.clone()
            with torch.no_grad():
                dt_remain = dt * torch.ones(len(X), device=X.device)
                cost = 0 * cost_fn(X, self.get_nominal_control(X))
                control_threshold = torch.tensor(self._control_threshold, device=X.device)
            while torch.max(dt_remain) > 0:
                indices_remain = dt_remain > 0
                X_remain = X_next[indices_remain]
                output = control_fn(X_next)[indices_remain] # for batchnorm
                U = self.get_nominal_control(X_remain) + output
                with torch.no_grad():
                    dt_scaled, _ = torch.min(dt * control_threshold / torch.abs(U), dim=1)
                    dt_new = torch.clamp(dt_scaled, max=dt_remain[indices_remain])
                    dt_remain = dt_remain.clone()
                    dt_remain[indices_remain] -= dt_new
                X_next = X_next.clone()
                X_next[indices_remain] += torch.diag(dt_new) @ self.get_xdot(X_remain, U)
                cost = cost.clone()
                cost[indices_remain] += torch.diag(dt_new) @ cost_fn(X_remain, output)
                
        elif step_type == 'RK4':
            xdot_fn = lambda _X: self.get_xdot(_X, self.get_nominal_control(_X) + control_fn(_X))
            k1 = xdot_fn(X)
            k2 = xdot_fn(X + (dt * k1 / 2))
            k3 = xdot_fn(X + (dt * k2 / 2))
            k4 = xdot_fn(X + (dt * k3))
            X_next = X + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            x_cost_fn = lambda _X: cost_fn(_X, control_fn(_X))
            cost_k1 = x_cost_fn(X)
            cost_k2 = x_cost_fn(X + (dt * k1 / 2))
            cost_k3 = x_cost_fn(X + (dt * k2 / 2))
            cost_k4 = x_cost_fn(X + (dt * k3))
            cost = dt * (cost_k1 + 2 * cost_k2 + 2 * cost_k3 + cost_k4) / 6
        else:
            raise NotImplementedError('Unsupported step type.')

        return X_next, cost

class Unicycle_Acc(ControlAffineSystem):
    """
    Unicycle system (x_coord, y_coord, angle, linear velocity, angular velocity)
    controlled by (linear acceleration, angular acceleration)
    """
    def __init__(self, init_box, l=0.1, radius=0.1, kappa=5):
        self._state_dim = 5
        self._encoded_state_dim = 5 + 1 # angle -> (sin, cos)
        self._control_dim = 2
        self._init_box = init_box
        self._l = l
        self._radius = radius
        self._kappa = kappa
        self._control_threshold = [100, 100]

    def get_center(self, X):
        C = torch.empty(X.shape[0], 2, dtype=X.dtype, device=X.device)
        px, py, theta = [X[:,i] for i in range(3)]
        C[:,0] = px + self._l * torch.cos(theta)
        C[:,1] = py + self._l * torch.sin(theta)
        return C
    
    def encode_input(self, X):
        """angle -> (sin, cos)"""
        theta = X[:,2:3]
        return torch.cat((X[:,:2], torch.cos(theta), torch.sin(theta), X[:,3:]), dim=1)

    def get_f(self, X):
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        return torch.stack([
            v * torch.cos(theta),
            v * torch.sin(theta),
            w,
            torch.zeros(X.shape[0], dtype=X.dtype, device=X.device),
            torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
            ], dim=1)

    def get_g(self, X):
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        G = torch.zeros(X.shape[0], X.shape[1], self.control_dim, dtype=X.dtype, device=X.device)
        G[:,3,0] = 1
        G[:,4,1] = 1
        return G
    
    def check_collision(self, X, obs):
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        del_x = px + self._l * torch.cos(theta) - obs.cx
        del_y = py + self._l * torch.sin(theta) - obs.cy
        return (del_x/obs.rx)**2 + (del_y/obs.ry)**2 < 1 - 1e-3
    
    def get_cbf_h(self, X, obs):
        """higher order CBF preventing collision from elliptical obstacle"""
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        del_x = px + self._l * cos_theta - obs.cx
        del_y = py + self._l * sin_theta - obs.cy
        del_x_dot = v * cos_theta - self._l * sin_theta * w 
        del_y_dot = v * sin_theta + self._l * cos_theta * w
        h_ellipse = (del_x/obs.rx)**2 + (del_y/obs.ry)**2 - 1.0
        h_ellipse_dot = 2* del_x * del_x_dot /(obs.rx**2) + 2* del_y * del_y_dot /(obs.ry**2)
        return h_ellipse_dot + self._kappa * h_ellipse
    
    def get_cbf_h_grad(self, X, obs):
        """gradient of CBF"""
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        del_x = px + self._l * cos_theta - obs.cx
        del_y = py + self._l * sin_theta - obs.cy
        del_x_dot = v * cos_theta - self._l * sin_theta * w 
        del_y_dot = v * sin_theta + self._l * cos_theta * w
        return torch.stack([
            2* del_x_dot /(obs.rx**2) + self._kappa * 2* del_x /(obs.rx**2),
            2* del_y_dot /(obs.ry**2) + self._kappa * 2* del_y /(obs.ry**2),
            (-2* self._l * sin_theta * del_x_dot -2* del_x * del_y_dot) /(obs.rx**2)
                + (2* self._l * cos_theta * del_y_dot +2* del_y * del_x_dot) /(obs.ry**2)
                - 2* self._kappa * del_x * self._l * sin_theta /(obs.rx**2)
                +2* self._kappa * del_y * self._l * cos_theta /(obs.ry**2),
            2* del_x * cos_theta /(obs.rx**2) + 2* del_y * sin_theta /(obs.ry**2),
            - 2* del_x * self._l * sin_theta /(obs.rx**2) + 2* del_y * self._l * cos_theta /(obs.ry**2)
        ], dim=1)

    def get_nominal_control(self, X):
        px, py, theta, v, w = [X[:,i] for i in range(X.shape[1])]
        d = torch.sqrt(px**2 + py**2)
        w_goal = 10 * (px * torch.sin(theta) - py * torch.cos(theta)) / (d+1e-5)
        v_lim = 5
        v_goal = torch.clamp(10 * d * (-px * torch.cos(theta) -py * torch.sin(theta)), -v_lim, v_lim)
        acc_lin = 5*(v_goal - v)
        acc_ang = 5*(w_goal - w)
        return torch.stack([acc_lin, acc_ang], dim=1)


class SafeControl:
    """ 
    Learn safe control from obstacles using Control barrier function (CBF)
    """
    def __init__(self, Q, R, X, sys, obs_list, loss_max, alpha, T=2, dt=0.05, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._R = torch.tensor(R)
        self._X = torch.tensor(X)
        self._Y = torch.zeros(X.shape[0], 0) # unsupervised learning
        self._sys = sys
        self._obs_list = obs_list
        self._loss_max = loss_max
        self._alpha = alpha
        self._dt = dt
        self._nstep = int(T / dt)
        self._encoded_xdim = sys.encoded_state_dim
        self._ydim = sys.control_dim
        self._partial_vars = np.arange(self._ydim)
        self._num = X.shape[0]
        self._neq = 0
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        ### For Pytorch
        self._device = None

    def __str__(self):
        return f'SafeControl-{self.num}'

    @property
    def Q(self):
        return self._Q

    @property
    def R(self):
        return self._R

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
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def R_np(self):
        return self.R.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def sys(self):
        return self._sys
    
    @property
    def obs_list(self):
        return self._obs_list

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
        return self._sys.encode_input(X)
    
    def get_nominal_control(self, X):
        return self._sys.get_nominal_control(X)

    def evaluate(self, X, Y):
        U = Y + self.get_nominal_control(X)
        return ((X@self.Q)*X).sum(dim=1) + ((U@self.R)*U).sum(dim=1)

    def get_resid(self, X, Y):
        A, bl, bu = self.get_coefficients(X)
        return torch.clamp((A@Y[:,:,None])[:,:,0] - bu, 0)
    
    def run_episode(self, net, X, accum_fn, nstep=None, dt=None, isTest=False, saveTraj=False, accum_max=None):
        """run one episode of simulation while accumulating cost up to accum_max"""
        nstep = self._nstep if nstep is None else nstep
        dt = self._dt if dt is None else dt
        traj = None
        X_next = X.clone()
        with torch.no_grad():
            accum_total = 0 * accum_fn(X, net(X))
            indicies_to_update = torch.ones(len(accum_total)).bool()
            if accum_max is not None:
                assert(len(accum_total.shape) == 1)
        if saveTraj:
            traj = torch.empty(nstep+1, X.shape[0], X.shape[1], dtype=X.dtype, device=X.device)
            traj[0,:,:] = X_next
        for i in range(nstep):
            control_fn = lambda _X: net(_X, isTest=isTest)
            X_next, accum_step = self._sys.step(X_next, dt, control_fn, accum_fn)
            if saveTraj:
                traj[i+1,:,:] = X_next
            accum_total = accum_total.clone()
            accum_total[indicies_to_update] += accum_step[indicies_to_update]
            if accum_max is not None:
                indicies_to_update = accum_total < accum_max
                if torch.any(indicies_to_update) == False:
                    break
        return accum_total, traj

    def get_train_loss_step(self, X, Y, args):
        main_loss = self.evaluate(X, Y)
        regularization = torch.norm(self.get_resid(X, Y), dim=1)**2
        return main_loss + args['softWeight'] * regularization
    
    def get_train_loss(self, net, X, Ytarget, args):
        accum_fn = lambda X, Y: self.get_train_loss_step(X, Y, args)
        return self.run_episode(net, X, accum_fn, accum_max=self._loss_max)[0]
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return self.run_episode(net, X, self.evaluate, isTest=True)[0]
    
    def get_err_metric1(self, net, X, Y, Ytarget):
        return self.run_episode(net, X, self.get_resid, isTest=True)[0]
    
    def get_err_metric2(self, net, X, Y, Ytarget):
        return torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

    def get_coefficients(self, X):
        """coefficients for inequality constraints bl(x)<=A_eff(x)y<=bu(x)"""
        h_list = []
        h_grad_list = []
        for obs in self._obs_list:
            h = self._sys.get_cbf_h(X, obs)
            h_list.append(h[:,None])
            h_grad = self._sys.get_cbf_h_grad(X, obs)
            h_grad_list.append(h_grad[:,None,:])
        h_concat = torch.cat(h_list, dim=1)
        h_grad_concat = torch.cat(h_grad_list, dim=1)

        g = self._sys.get_g(X)
        A = - h_grad_concat @ g
        bu = (h_grad_concat @ (g @ self.get_nominal_control(X)[:,:,None]\
                                + self._sys.get_f(X)[:,:,None]))[:,:,0]\
            + self._alpha * h_concat
        largeNum = 1e10
        bl = torch.zeros(X.shape[0], bu.shape[1], device=self.device) - largeNum
        return A, bl, bu

    ##### For DC3 #####

    def get_resid_grad(self, X, Y):
        A, bl, bu = self.get_coefficients(X)
        resid = torch.clamp((A@Y[:,:,None])[:,:,0] - bu, 0)
        return 2*(resid[:,None,:]@A)[:,0,:]
    
    def get_ineq_partial_grad(self, X, Y):
        return NotImplementedError

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        """no completion for this problem (Y=Z)"""
        return Z