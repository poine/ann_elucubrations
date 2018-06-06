#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time, logging, numpy as np, matplotlib.pyplot as plt
import control
import gym, gym_foo.envs.pvtol_pole as pvtp, utils as ut
LOG = logging.getLogger('test_gym_pvtol_pole')

'''
   Simulate the PVTOL pole system
'''

class StateFeedbackReg:
    def __init__(self, env, _kind='place'):
        self.Ue = env.pvtp.Ue
        A, B = env.pvtp.jac()
        if _kind == 'place':
            poles=[-5, -5, -5, -5, -5, -5, -5, -5]
            self.K = control.matlab.place(A, B, poles)
        else:
            Q = np.diag([1., 10., 1.5, 0.25, 0.01, 0.05, 0.005, 0.001])
            R = np.diag([0.5, 0.5])
            (self.K, X, E) = control.matlab.lqr(A, B, Q, R)
        poles, vect_p = np.linalg.eig(A-np.dot(B, self.K))
        LOG.info('gain:\n{}'.format(self.K))
        LOG.info('poles:\n{}'.format(poles))
            
    def predict(self, state, Xsp):
        dX = state - Xsp
        return (self.Ue - np.dot(self.K, dX)).squeeze()

def sim(env, r):
    _dt, _len = 0.01, 1200
    time = np.linspace(0., _len*_dt, _len)
    X, U, Xsp = np.zeros((_len, env.pvtp.s_size)), np.zeros((_len, env.pvtp.i_size)), np.zeros((_len, env.pvtp.s_size))
    Xsp[:,pvtp.PVTP.s_x] = ut.step_vec(time, dt=6, t0=2.7)
    X0 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    X[0] = env.reset(X0)[:env.pvtp.s_size]
    for i in range(1, len(time)):
        U[i-1] = r.predict(X[i-1], Xsp[i-1])
        state, reward, over, info = env.step(U[i-1])
        X[i] = state[:env.pvtp.s_size]
    U[-1] = U[-2]
    return time, X, U
    
def plot(time, X, U, figure=None, filename=None):
    if figure is None:
        figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=(0.04, 0.04, 0.98, 0.96, 0.13, 0.25))
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
    ut.save_if(filename)
    return figure
    
import pyglet
def render(env, time, X, U, fps=25, save=True):
    dt_frame, i_frame = 1./fps, 0
    for i in range(len(time)):
        if time[i]>=i_frame*dt_frame:
            env.pvtp.state = X[i]
            env.render(info='t={:.1f}s'.format(time[i]))
            if save:
                pyglet.image.get_buffer_manager().get_color_buffer().save('/tmp/sc/screenshot_{:04d}.png'.format(i_frame))
            i_frame += 1
    if save:
        os.system('apngasm /tmp/pvtol_pole_sim_1_anim.apng /tmp/sc/screenshot_0* 1 {:d}'.format(fps))

def main(_plot=True, _render=True):
    env_name = 'pvtol_pole-v0'
    env = gym.make(env_name)
    env.seed(123)
    r = StateFeedbackReg(env, 'lqr')
    time, X, U = sim(env, r)
    if _plot:
        plot(time, X, U, filename='/tmp/pvtol_pole_sim.png')
        plt.show()
    if _render:
        render(env, time, X, U)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    main(_plot='plot' in sys.argv, _render= 'render' in sys.argv)
