#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, signal, os, time, logging, numpy as np, matplotlib.pyplot as plt
import gym, gym_foo
import pdb

def signal_handler(signal, frame):
    print('sigint caught')
    sys.exit(0)

def test_1():
    env = gym.make('pendulum-ros-v0')
    time.sleep(0.1)
    env.reset()
    time.sleep(10)
    for i in range(1000):
        #env.step(np.array([0.]))
        env.step(np.array([env.max_torque/2]))
        time.sleep(env._dt)

def test_2():
    env = gym.make('pendulum-legacy-v0')
    env.reset()
    for i in range(1000):
        #env.step(np.array([0.]))
        env.step(np.array([env.max_torque]))
        env.render()
        time.sleep(env.dt)
    

def test_3():
    env = gym.make('doublependulum-ros-v0')
    time.sleep(0.1)
    env.reset()
    
    n_step, X = 100, []
    for i in range(n_step):
        Xi, costi, over, info = env.step(np.array([0.]))
        X.append(Xi)
        time.sleep(env._dt)
    X = np.array(X)
    _time = np.arange(0, n_step*env._dt, env._dt)
    _a1 = np.arctan2(X[:,1], X[:,0])
    _v1 = X[:,2]
    _a2 = np.arctan2(X[:,4], X[:,3])
    _v2 = X[:,5]
    plt.subplot(4, 1, 1)
    plt.plot(_time, _a1, '.')
    plt.subplot(4, 1, 2)
    plt.plot(_time, _v1, '.')
    plt.subplot(4, 1, 3)
    plt.plot(_time, _a2, '.')
    plt.subplot(4, 1, 4)
    plt.plot(_time, _v2, '.')
    plt.show()
    pdb.set_trace()
        
        
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    test_3()
