#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Dynamic model of a robot arm
'''

import logging, numpy as np, math, scipy.integrate, matplotlib.pyplot as plt

import pdb
import utils as ut

LOG = logging.getLogger('robot_arm')

#
# Parameters
#
class Param:
    def __init__(self, sat=None):
        self.l = 1.0  # length (in m)
        self.g = 9.81 # gravity

        self.a = 9.81 # spring
        self.b = 0.25 # friction
        self.c = 1.
        

class Plant:
    # State
    # phi:   arm position, (rad)
    # phid:  arm angular velocity, (rad/s)
    s_phi, s_phid, s_size = range(3)

    # input
    # tau: motor torque, (N.m)
    i_tau, i_size = range(2)

    def __init__(self, dt=0.01, P=None):
        self.dt = dt
        self.P = Param() if P is None else P
        P = self.P
        
    def cont_dyn(self, X, t, U):
        phi_dot = X[self.s_phid]
        phi_dot_dot = -self.P.a*np.sin(X[self.s_phi]) - self.P.b*X[self.s_phid] + self.P.c*U[self.i_tau]
        return np.array([phi_dot, phi_dot_dot])

    def disc_dyn(self, Xk, Uk):
        _unused, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [0, self.dt], args=(Uk,))
        return Xkp1

    def jacobian(self, Xe, Ue):
        A = np.array([[0, 1],[-self.P.a*np.cos(Xe[0]), - self.P.b]])
        B = np.array([[0],[self.P.c]])
        return A, B
    
    def sim(self, time, X0, ctl, pert=None):
        X, U = np.zeros((len(time), self.s_size)),  np.zeros((len(time), self.i_size))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], time[i-1], i-1)
            X[i] = self.disc_dyn(X[i-1], U[i-1]+(pert[i] if pert is not None else 0))
        U[-1] = U[-2]
        return X, U





def plot(time, X, U=None, sp=None, ref=None, figure=None):
    margins = (0.08, 0.1, 0.95, 0.93, 0.2, 0.46)#left, bottom, right, top, wspace, hspace
    figure = ut.prepare_fig(figure, figsize=(10.24, 7.68), margins=margins)
    ax = plt.subplot(3, 1, 1)
    plt.plot(time, ut.deg_of_rad(X[:,Plant.s_phi]))
    if sp is not None: plt.plot(time, ut.deg_of_rad(sp[:,0]))
    if ref is not None: plt.plot(time, ut.deg_of_rad(ref[:,0]))
    ut.decorate(ax, title="$\\theta$", ylab='deg')
    ax = plt.subplot(3, 1, 2)
    plt.plot(time, ut.deg_of_rad(X[:,Plant.s_phid]))
    if ref is not None: plt.plot(time, ut.deg_of_rad(ref[:,1]))
    ut.decorate(ax, title="$\dot{\\theta}$", ylab='deg/s')
    ax = plt.subplot(3, 1, 3)
    if U is not None:
        plt.plot(time, U[:,Plant.i_tau])
        ut.decorate(ax, title="$\\tau$", ylab='N.m', xlab='time in s')
    return figure


def make_or_load_training_set(plant, make_training_set, filename, nsamples=int(50*1e3)):
    if make_training_set:
        dt, max_nperiod = 0.01, 10
        LOG.info('  Generating random setpoints')
        if 1:
            time, yc = ut.make_random_pulses(plant.dt, nsamples, max_nperiod=max_nperiod,  min_intensity=-4, max_intensity=4.)
        else:
            time = np.arange(0, nsamples*dt, dt)
            yc = ut.sine_swipe_input_vec(time)
        LOG.info('   done. Generated {} random setpoints'.format(len(time)))
        LOG.info('  Simulating trajectory ({} s)'.format(time[-1]))
        def ctl(X,t, k): return [yc[k]]
        X0 = [0., 0.]
        X, U = plant.sim(time, X0, ctl)
        LOG.info('   done')
        LOG.info('  Saving trajectory to {}'.format(filename))
        desc = 'random setpoint trajectory. max_nperiod: {}'.format(max_nperiod)
        ut.save_trajectory(time, X, U, desc, filename)
    else:
        LOG.info('  Loading trajectory from {}'.format(filename))
        time, X, U, desc = ut.load_trajectory(filename)
        LOG.info('     {} samples ({} s)'.format(len(time), time[-1]))
        LOG.info('     desc: {}'.format(desc))
        
    return time, X, U, desc    






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    plant = Plant()
    time =  np.arange(0., 15.05, plant.dt)
    yc = ut.step_vec(time, dt=8)
    def ctl(X,t, k): return [yc[k]]
    X0 = [0.1, 0.1]
    X, U = plant.sim(time, X0, ctl)
    plot(time, X, U)
    plt.savefig('../docs/images/robot_arm_free_trajectory.png')
    plt.show()
