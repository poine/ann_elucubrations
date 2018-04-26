#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Dynamic (kinematic) model of a bicycle
'''

import numpy as np, math, scipy.integrate, matplotlib.pyplot as plt
import control.matlab
import pdb
#import pure_pursuit, guidance, guidance.utils as gut, guidance.path_factory
import two_d_guidance, utils as ut

#
# Parameters
#
class Param:
    def __init__(self, sat=None):
        self.LR, self.LF = 0.05, 0.05
        self.L = self.LR+self.LF
        self.alpha = self.LR/self.L

#
# Components of the state vector
#
s_x      = 0  # x position of the CG in m
s_y      = 1  # y position of the CG in m
s_psi    = 2  # orientation of the body in rad, 0 up
s_v      = 3  # horizontal velocity of the CG in m/s
s_size   = 4  # dimension of the state vector

#
# Components of the input vector
#
i_a    = 0  # longitudinal acceleration in m/s2
i_df   = 1  # orientation of front wheel in rad
i_size = 2  # dimension of the input vector

# Dynamic model as continuous time state space representation
#
# X : state
# U : input
# P : param
#
# returns Xd, time derivative of state
#
def dyn(X, t, U, P):
    beta = np.arctan(P.alpha*U[i_df])
    th = X[s_psi]+beta
    sth, cth = np.sin(th), np.cos(th)
    xd = X[s_v]*cth
    yd = X[s_v]*sth
    psid = X[s_v]/P.LR*np.sin(beta)
    vd = U[i_a]
    return np.array([xd, yd, psid, vd])

def disc_dyn(Xk, Uk, dt, P):
    Xkp1 =  scipy.integrate.odeint(dyn, Xk, [0, dt], args=(Uk, P ))[1]
    return Xkp1

class Plant:
    def __init__(self, P=None):
        self.P = P if P is not None else Param()

    def disc_dyn(self, Xk, Uk, dt):
        return disc_dyn(Xk, Uk, dt, self.P)

def plot(time, X, U, Y=None, figure=None, window_title="trajectory"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = ut.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    plots = [("x", "m", X[:,s_x]),
             ("y", "m", X[:,s_y]),
             ("$\psi$", "deg", ut.deg_of_rad(X[:,s_psi])),
             ("$v$", "m/s", X[:,s_v]),
             ("$\delta_f$", "deg", ut.deg_of_rad(U[:,i_df]))]
    for i, (title, ylab, data) in enumerate(plots):
        ax = plt.subplot(len(plots), 1, i+1)
        plt.plot(time, data, linewidth=2)
        ut.decorate(ax, title=title, ylab=ylab)
    return figure

def plot2D(time, X, U, Y=None, figure=None, window_title="trajectory"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = ut.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    plt.plot(X[:,s_x], X[:,s_y])
    plt.plot(Y[:,s_x], Y[:,s_y], '--')
    plt.legend(['sys','ref'])
    plt.axes().set_aspect('equal')

def sim_open_loop(p, X0, df0=0.):
    U = [0, df0]
    time = np.arange(0., 10, 0.01)
    X = scipy.integrate.odeint(dyn, X0, time, args=(U, p ))
    U = np.zeros((len(time), i_size))
    psi0, v0 = X0[s_psi], X0[s_v]
    p0 = X0[:s_psi]; _v0 = v0*np.array([math.cos(psi0), math.sin(psi0)])
    if df0 == 0.:
        p1 = p0 + time[-1]*_v0
        _path = two_d_guidance.path_factory.make_line_path(p0, p1)
    else:
        rad = (p.LR+p.LF)/math.tan(df0)
        c = p0 + rad*np.array([-_v0[1], _v0[0]])
        _path = two_d_guidance.path_factory.make_circle_path(c, rad, 0, 2*math.pi, n_pt=100)
    #pdb.set_trace()
    return time, X, U, _path.points

def sim_pure_pursuit(p, X0, path_filename, duration=10.):
    ctl =  two_d_guidance.PurePursuit(path_filename, p)
    time = np.arange(0., duration, 0.01)
    X, U = np.zeros((len(time), s_size)), np.zeros((len(time), i_size))
    X[0] = X0
    for i in range(1, len(time)):
        try:
            U[i-1] = ctl.compute([X[i-1,s_x], X[i-1,s_y]], X[i-1,s_psi])
        except two_d_guidance.EndOfPathException:
            ctl.path.reset()
            U[i-1] = ctl.compute([X[i-1,s_x], X[i-1,s_y]], X[i-1,s_psi])
        X[i] =  scipy.integrate.odeint(dyn, X[i-1], [time[i-1], time[i]], args=(U[i-1], p ))[1]
        X[i, s_psi] = two_d_guidance.norm_yaw(X[i, s_psi])
    U[-1] = U[-2]
    return time, X, U, ctl.path.points

if __name__ == "__main__":
    p = Param()
    #X0= [1.2, 0.45, 0., 1]
    X0= [0., 3.3, 0., 1]
    #time, X, U, R = sim_open_loop(p, X0, 0.2)
    path_file='/home/poine/work/ann_elucubrations/data/paths/oval_01.npz'
    path_file='/home/poine/work/ann_elucubrations/data/paths/track_ethz_dual_01.npz'
    time, X, U, R = sim_pure_pursuit(p, X0, path_file, 23.)
    
    plot(time, X, U)
    plot2D(time, X, U, R)
    plt.show()
