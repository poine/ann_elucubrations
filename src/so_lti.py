#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
Dynamic model of a second order Linear Time Invariant System
'''


import logging, timeit, math, numpy as np, scipy.signal, scipy.integrate, matplotlib.pyplot as plt, pickle

LOG = logging.getLogger('so_lti')

import utils as ut

class Plant:
    def __init__(self, Ac, Bc, dt):
        self.Ac, self.Bc, self.dt = Ac, Bc, dt
        LOG.debug('Ac\n{}\nBc\n{}'.format(self.Ac, self.Bc))
        self.Ad = scipy.linalg.expm(dt*self.Ac)
        tmp = np.dot(np.linalg.inv(self.Ac), self.Ad-np.eye(2))
        self.Bd = np.dot(tmp, self.Bc)
        LOG.info('\nAd\n{}\nBd\n{}'.format(self.Ad, self.Bd))

    def cont_dyn(self, X, t, U):
        Xd = np.dot(self.Ac, X) + np.dot(self.Bc, U)
        return Xd

    def disc_dyn(self, Xk, Uk):
        Xkp1 = np.dot(self.Ad,Xk) + np.dot(self.Bd, Uk)
        return Xkp1

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 2)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.disc_dyn(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U
    
class CCPlant(Plant):
    ''' Control Companion form '''
    def __init__(self, omega=1, xi=0.9, dt=0.01):
        self.omega, self.xi = omega, xi
        Ac = np.array([[0, 1],[-omega**2, -2*xi*omega]])
        Bc = np.array([[0],[omega**2]])
        Plant.__init__(self, Ac, Bc, dt)




def plot(time, X, U=None, Yc=None):
    ax = plt.subplot(3,1,1)
    plt.plot(time, X[:,0])
    if Yc is not None: plt.plot(time, Yc, 'k')
    ut.decorate(ax, title="$x_1$", ylab='time')
    ax = plt.subplot(3,1,2)
    plt.plot(time, X[:,1])
    ut.decorate(ax, title="$x_2$", ylab='time')
    if U is not None:
        ax = plt.subplot(3,1,3)
        plt.plot(time, U)
        ut.decorate(ax, title="$u$", ylab='time')



def make_or_load_training_set(plant, ctl, make_training_set, filename = '/tmp/so_lti_training_traj.pkl'):
    if make_training_set:
        nsamples, max_nperiod = int(10*1e3), 10
        LOG.info('  Generating random setpoints')
        time, ctl.yc = ut.make_random_pulses(plant.dt, nsamples, max_nperiod=max_nperiod,  min_intensity=-10, max_intensity=10.)
        LOG.info('   done. Generated {} random setpoints'.format(len(time)))
        LOG.info('  Simulating trajectory ({} s)'.format(time[-1]))
        X0 = [0.]
        X, U = plant.sim(time, X0, ctl.get)
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
    plant = CCPlant(omega=2, xi=0.7)
    ctl = ut.CtlNone()
    time =  np.arange(0., 15.05, plant.dt)
    ctl.yc = ut.step_input_vec(time, dt=8)
    X0 = [0, 0]
    X1, U1 = plant.sim(time, X0, ctl.get)
    plot(time, X1, U1)
    plt.show()
