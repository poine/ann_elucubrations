#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Dynamic model of a bicycle
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
        self.Lr, self.Lf = 0.05, 0.05 # rear and front axle positions wrt center of mass (in m)
        self.m = 0.2                  # mass (in kg)
        self.J = 0.003                # inertia 
        self.g = 9.81                 # gravity (in m/s2)
        self.mu = 1.75                # road-tire friction (TBD, eq 3.5)
        self.pacejka = pacejka_paper  # tire model
        self.compute_aux()
        
    def compute_aux(self):
        self.L = self.Lr+self.Lf             # axles distance (in m)
        self.cgp = self.Lr/self.L            # relative position of center of gravity (unitless)
        self.fc = -0.5*self.m*self.g*self.mu # force coefficient
        
#
# Components of the state vector
#
s_x      = 0  # x position of the CG in m
s_y      = 1  # y position of the CG in m
s_psi    = 2  # orientation of the body in rad, 0 up
s_vx     = 3  # body-fixed x axis velocity in m/s
s_vy     = 4  # body-fixed y axis velocity in m/s
s_psid   = 5  # yaw rate in rad/s
s_size   = 6  # dimension of the state vector

#
# Components of the input vector
#
i_a    = 0  # longitudinal acceleration in m/s2
i_df   = 1  # orientation of front wheel in rad
i_size = 2  # dimension of the input vector

# tire force model
def pacejka(alpha, B, C, D): return D*np.sin(C*np.arctan(B*alpha))

def pacejka_dry(alpha):  return pacejka(alpha, 10., 1.9, 1.)
def pacejka_snow(alpha): return pacejka(alpha,  5., 2., 0.3)
def pacejka_paper(alpha): return pacejka(alpha, 2., 2., 0.5)

# Dynamic model as continuous time state space representation
#
# X : state
# U : input
# P : param
#
# returns Xd, time derivative of state
#
def dyn(X, t, U, P):
    Xd = np.zeros(s_size)

    spsi, cpsi = math.sin(X[s_psi]), math.cos(X[s_psi])
    Xd[s_x] = cpsi*X[s_vx] - spsi*X[s_vy] # velocity in world frame
    Xd[s_y] = spsi*X[s_vx] + cpsi*X[s_vy]
    Xd[s_psi] = X[s_psid]

    # what do I do when vel is 0?....
    abs_vx = math.fabs(X[s_vx])
    if abs_vx > 1e-12:
        alpha_f = math.atan((X[s_vy]+P.Lf*X[s_psid])/abs_vx) - U[i_df]
        alpha_r = math.atan((X[s_vy]-P.Lr*X[s_psid])/abs_vx)
        Ff, Fr = P.fc*P.pacejka(alpha_f), P.fc*P.pacejka(alpha_r)
    else:
        Ff, Fr = 0.,0.
    Xd[s_vx] = U[i_a] + X[s_psid]*X[s_vy]
    Xd[s_vy] = 1/P.m*(Ff*math.cos(U[i_df])+Fr) - X[s_psid]*X[s_vx]
    Xd[s_psid] = 1/P.J*(P.Lf*Ff-P.Lr*Fr)

    return Xd

def disc_dyn(Xk, Uk, dt, P):
    return scipy.integrate.odeint(dyn, Xk, [0, dt], args=(Uk, P ))[1]




class Plant:
    def __init__(self, P=None):
        self.P = P if P is not None else Param()

    def disc_dyn(self, Xk, Uk, dt):
        return disc_dyn(Xk, Uk, dt, self.P)



def plot_friction():
    alphas = np.linspace(ut.rad_of_deg(-30), ut.rad_of_deg(30), 100)
    F1 = pacejka_dry(alphas)
    F2 = pacejka_snow(alphas)
    plt.plot(ut.deg_of_rad(alphas), F1)
    plt.plot(ut.deg_of_rad(alphas), F2)
    ut.decorate(plt.gca(), 'tire friction', 'degres', 'unitless', ['dry','snow'])
    plt.show()

def plot_time(time, X, U, Y=None, figure=None, window_title="trajectory"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = ut.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    if 0:
        plots = [("vx", "m/s", X[:,s_vx]),
                 ("vy", "m/s", X[:,s_vy]),
                 ("a", "m/s2", U[:,i_a]),
                 ("df", "deg", ut.deg_of_rad(U[:,i_df]))]
    else:
        
        v = np.linalg.norm(X[:,s_vx:s_vy+1], axis=1)
        plots = [("$\psi$", "deg", ut.deg_of_rad(X[:,s_psi])),
                 ("$v$", "m/s", v),
                 ("$\dot{\psi}$", "deg/s", ut.deg_of_rad(X[:,s_psid])),
                 ("$a$", "m/s2", U[:,i_a]),
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
    if Y is not None: plt.plot(Y[:,s_x], Y[:,s_y], '--')
    plt.legend(['sys','ref'])
    plt.axes().set_aspect('equal')
    return figure


def sim_open_loop(p, X0, df0=0.):
    U = [0, df0]
    time = np.arange(0., 10, 0.01)
    X = scipy.integrate.odeint(dyn, X0, time, args=(U, p ))
    U = U*np.ones((len(time), i_size))
    return time, X, U, None


def sim_pure_pursuit(p, X0, path_filename, duration=44, v_sp=0.9, stop_at_eop=False):
    ctl =  two_d_guidance.PurePursuit(path_filename, p)
    #path_len = ctl.path.dists[-1]
    v_ctl = two_d_guidance.VelController(v_sp=v_sp)
    time = np.arange(0., duration, 0.01)
    _len = len(time)
    X, U = np.zeros((len(time), s_size)), np.zeros((len(time), i_size))
    X[0] = X0
    for i in range(1, len(time)):
        try:
            U[i-1] = ctl.compute([X[i-1,s_x], X[i-1,s_y]], X[i-1,s_psi])
        except two_d_guidance.EndOfPathException:
            print('finished at {}s'.format(time[i]));
            if stop_at_eop:
                _len=i;  break
            else:
                ctl.path.reset()
                U[i-1] = ctl.compute([X[i-1,s_x], X[i-1,s_y]], X[i-1,s_psi])
        U[i-1,i_a] = v_ctl.compute(X[i-1,s_vx])
        X[i] =  disc_dyn(X[i-1], time[i]-time[i-1], U[i-1], p )
        X[i, s_psi] = two_d_guidance.norm_yaw(X[i, s_psi])
    U[-1] = U[-2]
    return time[:_len], X[:_len], U[:_len], ctl.path.points


if __name__ == "__main__":

    #plot_friction()
    p = Param()
    #X0 = [1.2, 0.45, 0, 1., 0, 0]
    #time, X, U, R = sim_open_loop(p, X0, 0.2)
    #time, X, U, R = sim_pure_pursuit(p, X0, '/home/poine/work/oscar.git/oscar/oscar_control/paths/foh_01.npz')
    X0= [-.2, 3.2, 0., 0.01, 0, 0]
    time, X, U, R = sim_pure_pursuit(p, X0, '/home/poine/work/oscar.git/oscar/oscar_control/paths/track_ethz_dual_01.npz',
                                     v_sp=0.5, stop_at_eop=True)
    plot_time(time, X, U, R)
    plot2D(time, X, U, R)
    plt.show()
