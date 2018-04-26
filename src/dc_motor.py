#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
  Dynamic model of a DC motor
'''


import logging, timeit, math, numpy as np, scipy.signal, scipy.integrate, matplotlib.pyplot as plt, pickle, os
import control

LOG = logging.getLogger('dc_motor')

import utils as ut

import pdb

# Params
class Param:
    def __init__(self):
        self.Ra =  1.      # armature resistance, (ohm)
        self.La =  0.5     # armature inductance, (H)
        self.J =   0.01    # moment of inertia of the motor rotor and load, (Kg.m2/s2)
        self.B =   0.1     # damping ratio of the mechanical system, (Nms)
        self.Kv =  0.01    # back EMF factor  (V/rad/sec)
        self.Kt =  0.01    # torque factor constant, (Nm/Amp)

        
class Plant:
    # State
    # phi: The shaft position, (rad)
    # om:  The speed of the shaft and the load (angular velocity), (rad/s)
    # ia:  The armature current (Amp)
    s_phi, s_om, s_ia, s_size = range(4)

    # input
    # Va: input voltage
    # Tl: load torque
    i_va, i_tl, i_size = range(3)

    def __init__(self, dt=0.01, P=None):
        self.dt = dt
        self.P = Param() if P is None else P
        P = self.P
        self.Ac = np.array([[0., 1, 0],[0, -P.B/P.J, P.Kt/P.J],[0, -P.Kv/P.La, -P.Ra/P.La]])
        self.Bc = np.array([[0, 0],[0, -1./P.J],[1./P.La, 0]])
        if 0: # compute discretized system
            self.Ad = scipy.linalg.expm(dt*self.Ac)
            #print np.linalg.eig(self.Ac)
            self.Bd = None
        else:
            ct_sys = control.ss(self.Ac, self.Bc, [[1, 0, 0]], [[0,0]])
            dt_sys = control.sample_system(ct_sys, self.dt, method='zoh') #  ‘matched’, ‘tustin’, ‘zoh’
            self.Ad, self.Bd = dt_sys.A, dt_sys.B 
        LOG.info('\nAd\n{}\nBd\n{}'.format(self.Ad, self.Bd))
        tf_disc = control.tf(dt_sys)
        print tf_disc
        
    def cont_dyn(self, X, t, U):
        phi_dot = X[self.s_om]
        om_dot = 1/self.P.J*( self.P.Kt*X[self.s_ia] - U[self.i_tl] - self.P.B*X[self.s_om])
        ia_dot = 1/self.P.La*( U[self.i_va] - self.P.Kv*X[self.s_om] - self.P.Ra*X[self.s_ia])
        return np.array([phi_dot, om_dot, ia_dot])

    def disc_dyn2(self, Xk, Uk):
        _unused, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [0, self.dt], args=(Uk,))
        return Xkp1

    def disc_dyn(self, Xk, Uk):
        return np.dot(self.Ad, Xk) + np.dot(self.Bd, Uk)
    
    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), self.s_size)),  np.zeros((len(time), self.i_size))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)#; U[i-1,1]=0
            X[i] = self.disc_dyn(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U



def plot(time, X, U):
    ax = plt.subplot(4, 1, 1)
    plt.plot(time, X[:,Plant.s_phi])
    ut.decorate(ax, title="$\\theta$", ylab='rad')
    ax = plt.subplot(4, 1, 2)
    plt.plot(time, X[:,Plant.s_om])
    ut.decorate(ax, title="$\omega$", ylab='rad/s')
    ax = plt.subplot(4, 1, 3)
    plt.plot(time, X[:,Plant.s_ia])
    ut.decorate(ax, title="$i$", ylab='A')
    ax = plt.subplot(4, 1, 4)
    plt.plot(time, U[:,Plant.i_va])
    ut.decorate(ax, title="$V$", ylab='V')



def make_or_load_training_set(plant, make_training_set, filename = '/tmp/dc_motor_training_traj.pkl', nsamples=int(10*1e3)):
    if make_training_set or not os.path.isfile(filename):
        dt, max_nperiod = 0.01, 10
        LOG.info('  Generating random setpoints')
        if 1:
            time, yc = ut.make_random_pulses(plant.dt, nsamples, max_nperiod=max_nperiod,  min_intensity=-10, max_intensity=10.)
        else:
            time = np.arange(0, nsamples*dt, dt)
            yc = ut.sine_swipe_input_vec(time)
        LOG.info('   done. Generated {} random setpoints'.format(len(time)))
        LOG.info('  Simulating trajectory ({} s)'.format(time[-1]))
        def ctl(X,k): return [yc[k], 0]
        X0 = [0., 0., 0.]
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
    yc = ut.step_input_vec(time, dt=8)
    def ctl(X,k): return [yc[k], 0]
    X0 = [0, 0, 0]
    X, U = plant.sim(time, X0, ctl)
    plot(time, X, U)
    plt.show()
