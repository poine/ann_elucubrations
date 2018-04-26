#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
Dynamic model of a second order Linear Time Invariant System
'''


import logging, timeit, math, numpy as np, scipy.signal, scipy.integrate, matplotlib.pyplot as plt, pickle

import control, pdb

LOG = logging.getLogger('so_lti')

import utils as ut

# controllability matrix
def Qc(A, B): return np.concatenate([B, np.dot(A, B)], axis=1)


class Plant:
    def __init__(self, Ac, Bc, dt):
        self.Ac, self.Bc, self.dt = Ac, Bc, dt
        LOG.debug('Ac\n{}\nBc\n{}'.format(self.Ac, self.Bc))
        self.Ad = scipy.linalg.expm(dt*self.Ac)
        tmp = np.dot(np.linalg.inv(self.Ac), self.Ad-np.eye(2))
        self.Bd = np.dot(tmp, self.Bc)
        LOG.debug('\nAd\n{}\nBd\n{}'.format(self.Ad, self.Bd))

    def analyse(self):
        eva, eve = np.linalg.eig(self.Ad)
        LOG.info('poles {}'.format(eva))
        car_pol = np.poly(eva)
        _Qc = Qc(self.Ad, self.Bd)
        controllable = np.linalg.matrix_rank(_Qc)
        
        if controllable: # let's compute a companion form
            Ad_cc = np.array([[0, 1],[-car_pol[2], -car_pol[1]]])
            Bd_cc = np.array([[0],[1]])
            _Qc_cc = Qc(Ad_cc, Bd_cc)
            self.invM = np.dot(_Qc_cc, np.linalg.inv(_Qc))
            # check
            self.M = np.linalg.inv(self.invM)
            np.allclose(Ad_cc, np.dot(self.invM, np.dot(self.Ad, self.M)))
            np.allclose(Bd_cc, np.dot(self.invM, self.Bd))
        
        ss_disc = control.ss(self.Ad, self.Bd, [[1,0]], [[0]], self.dt)
        tf_disc = control.tf(ss_disc)
        print tf_disc
        self.a1, self.a0 = tf_disc.num[0][0]
        self.b2, self.b1, self.b0 = tf_disc.den[0][0]
        #pdb.set_trace()
        
        
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

    def sim_io(self, time, X0, ctl):
        X, U = np.zeros((len(time), 2)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i,0] = -self.b1*X[i-1, 0]-self.b0*X[i-2, 0]+self.a1*U[i-1,0]+self.a0*U[i-2,0]
        U[-1] = U[-2]
        X[:,1] = float('nan')
        return X, U
    
    
class CCPlant(Plant):
    ''' Control Companion form '''
    def __init__(self, omega=1, xi=0.9, dt=0.01):
        self.omega, self.xi = omega, xi
        Ac = np.array([[0, 1],[-omega**2, -2*xi*omega]])
        Bc = np.array([[0],[omega**2]])
        Plant.__init__(self, Ac, Bc, dt)




def plot2(time, X, U=None, Yc=None, figure=None, window_title="trajectory"):
    margins=(0.04, 0.1, 0.98, 0.95, 0.20, 0.29)
    figure = ut.prepare_fig(figure, window_title, figsize=(0.5*20.48, 0.5*10.24), margins=margins)
    ax = plt.subplot(2,1,1)
    plt.plot(time, X[:,0])
    if Yc is not None: plt.plot(time, Yc, 'k')
    ut.decorate(ax, title="$y$")
    if U is not None:
        ax = plt.subplot(2,1,2)
        plt.plot(time, U)
        ut.decorate(ax, title="$u$", xlab='time')
    return figure
        
def plot(time, X, U=None, Yc=None):
    ax = plt.subplot(3,1,1)
    plt.plot(time, X[:,0])
    if Yc is not None: plt.plot(time, Yc[:,0], 'k')
    ut.decorate(ax, title="$x_1$")
    ax = plt.subplot(3,1,2)
    plt.plot(time, X[:,1])
    ut.decorate(ax, title="$x_2$")
    if U is not None:
        ax = plt.subplot(3,1,3)
        plt.plot(time, U)
        ut.decorate(ax, title="$u$", xlab='time')



def make_or_load_training_set(plant, ctl, make_training_set, filename = '/tmp/so_lti_training_traj.pkl', nsamples=int(10*1e3)):
    if make_training_set:
        max_nperiod =  10
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
