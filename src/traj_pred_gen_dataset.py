#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import logging, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt

import traj_pred_utils as tpu
#from traj_pred_utils import Trajectory

import pdb


def gen_dataset_circles(filename=None):
    ''' 
    coordinated turn:
    m.g = L.cos(phi)
    m.v^2/R = L.sin(phi)
    R = v^2/(g.tan(phi))
    '''
    d = tpu.DataSet()
    d.trajectories = []
    dt, g, v = 1., 9.81, 240.
    for phi in np.arange(-np.deg2rad(45.), np.deg2rad(45.), np.deg2rad(0.3)):
        t = tpu.Trajectory(None)
        t.time = np.arange(0, 360, dt)
        t.curv = g*np.tan(phi)/v**2*np.ones(len(t.time))
        psi_dot = g*np.tan(phi)/v # turn rate
        R = v**2/g/np.tan(phi)    # radius
        psis = psi_dot * t.time
        t.pos = R*np.array([[np.sin(psi), np.cos(psi)-1] for psi in psis])
        t.vel = psi_dot*R*np.array([[np.cos(psi), -np.sin(psi)] for psi in psis])
        d.trajectories.append(t)
    if filename is not None: d.save(filename)
    return d


def discretize(cont_pos, cont_vel):
    step_pos = tpu.m_of_NM(1./64)
    disc_pos = np.round(cont_pos / step_pos)*step_pos
    step_vel = tpu.mps_of_kt(1.)
    disc_vel = np.round(cont_vel / step_vel)*step_vel
    return disc_pos, disc_vel
    

def gen_dataset_circles_disc(filename=None):
    d = gen_dataset_circles()
    for t in d.trajectories:
        t.pos_true, t_vel_true = t.pos, t.vel
        t.pos, t.vel = discretize(t.pos, t.vel)
    if filename is not None: d.save(filename)
    return d
    
    
s_x, s_y, s_xd, s_yd, s_phi, s_phid, s_size = range(7)
om_phi, xi_phi = 1., 0.9
om2_phi, txiom_phi = om_phi**2, 2*xi_phi*om_phi
def dyn(X,t, phi_sp, v_sp, g):
    Xdot=np.zeros(6)
    Xdot[s_x:s_y+1] = X[s_xd:s_yd+1]
    Xdot[s_phi] = X[s_phid]
    an, psi = g*np.tan(X[s_phi]), np.arctan2(X[s_yd], X[s_xd])
    Xdot[s_xd:s_yd+1] = [-an*np.sin(psi), an*np.cos(psi)]
    Xdot[s_phid] = -txiom_phi*X[s_phid] - om2_phi*(X[s_phi]-phi_sp(t)) 
    
    return Xdot
    
def step_sp(t, a, _dt=30): return a if math.fmod(t, _dt) > _dt/2 else 0


def gen_dataset_dyn():
    d = tpu.DataSet()
    d.trajectories = []
    dt, g, v = 1., 9.81, 240.

    for phi_sp in np.arange(-np.deg2rad(45.), np.deg2rad(45.), np.deg2rad(0.3)):
        t = tpu.Trajectory(None)
        t.time = np.arange(0, 360, dt) 
        X0 = [0, 0, 240., 0, 0, 0]
        X = scipy.integrate.odeint(dyn, X0, t.time, args=(lambda t: step_sp(t, phi_sp), v, g))
        if 0:
            plt.plot(t.time, X[:,s_phi])
            #plt.plot(X[:,s_x], X[:,s_y])
            plt.show()
            #pdb.set_trace()
        t.pos = X[:,s_x:s_y+1]
        t.vel = X[:,s_xd:s_yd+1]
        t.curv = g/v**2*np.tan(X[:,s_phi]) 
        d.trajectories.append(t)
    d.save('../data/traj_pred/ds_dyn.pkl')

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #gen_dataset_circles('../data/traj_pred/ds_circles.pkl')
    gen_dataset_circles_disc('../data/traj_pred/ds_circles_disc.pkl')
    #gen_dataset_dyn()

