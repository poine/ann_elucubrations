#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, logging, numpy as np, matplotlib.pyplot as plt
import control
import gym, gym_foo.envs.pvtol_pole as pvtp, utils as ut
LOG = logging.getLogger('test_gym_pvtol_pole')

class StateFeedbackReg:
    def __init__(self, v):
        self.Ue = v.Ue
        A, B = v.jac()
        poles=[-5, -5, -5, -5, -5, -5, -5, -5]
        self.K = control.matlab.place(A, B, poles)
        
    def predict(self, state, Xsp):
        dX = state - Xsp
        return (self.Ue - np.dot(self.K, dX)).squeeze()

def sim(v, r):
    _dt, _len = 0.01, 900
    time = np.linspace(0., _len*_dt, _len)
    Xsp = np.zeros((_len, v.s_size))
    Xsp[:,pvtp.PVTP.s_x] = ut.step_vec(time, dt=6)
    X, U = np.zeros((_len, v.s_size)), np.zeros((_len, v.i_size))
    X[0] = np.zeros(8)#np.array(v.state)
    for i in range(1, len(time)):
        U[i-1] = r.predict(X[i-1], Xsp[i-1])
        X[i] = v.disc_dyn(X[i-1], U[i-1], _dt)
    U[-1] = U[-2]
    return time, X, U
    
def plot(time, X, U):
    ax = plt.subplot(3, 2, 1)
    plt.plot(time, X[:, pvtp.PVTP.s_x])
    ut.decorate(ax, title='$x$', ylab='m')
    ax = plt.subplot(3, 2, 2)
    plt.plot(time, X[:, pvtp.PVTP.s_z])
    ut.decorate(ax, title='$z$', ylab='m')
    ax = plt.subplot(3, 2, 3)
    plt.plot(time, np.rad2deg(X[:, pvtp.PVTP.s_th]))
    ut.decorate(ax, title='$\\theta$', ylab='deg')
    ax = plt.subplot(3, 2, 4)
    plt.plot(time, np.rad2deg(X[:, pvtp.PVTP.s_ph]))
    ut.decorate(ax, title='$\phi$', ylab='deg')
    ax = plt.subplot(3, 2, 5)
    plt.plot(time, U[:, pvtp.PVTP.i_f1])
    plt.plot(time, U[:, pvtp.PVTP.i_f2])
    ut.decorate(ax, title='$f$', ylab='N')

import pyglet
def render(env, time, X, U, fps=25., save=True):
    for i in range(len(time)):
        env.pvtp.state = X[i]
        env.render(info='t={:.1f}s'.format(time[i]))
        if save:
            pyglet.image.get_buffer_manager().get_color_buffer().save('/tmp/sc/screenshot_{:04d}.png'.format(i))
    if save:
        os.system('apngasm /tmp/pvtol_pole_sim_1_anim.apng /tmp/sc/screenshot_0*')

def main(_plot=True, _render=True):
    env_name = 'pvtol_pole-v0'
    env = gym.make(env_name)
    env.seed(123)
    v = env.pvtp
    r = StateFeedbackReg(v)
    time, X, U = sim(v, r)
    if _plot:
        plot(time, X, U)
        plt.show()
    if _render:
        render(env, time, X, U)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    main(_plot=False)
