#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time, numpy as np, gym
import dql_utils, gym_foo, control

import pdb

class StateFeedbackReg:
    def __init__(self, env):
        self.Ue = env.pvtol.mass*env.pvtol.gravity/2*np.ones((2,1))
        A, B = env.pvtol.jac()
        poles=[-5, -5, -5, -5, -5, -5]
        self.K = control.matlab.place(A, B, poles)
        #pdb.set_trace()
        #self.K = np.zeros((2,6))

    def predict(self, state):
        Xref = np.array([state[6], state[7], 0, 0, 0, 0])
        dX = (state[:6] - Xref).reshape((6,1))
        return (self.Ue - np.dot(self.K, dX)).squeeze()


if __name__ == '__main__':
    np.set_printoptions(linewidth=300, suppress=True)
    env_name = 'planar_quad-v0'
    env = gym.make(env_name)
    env.seed(123)

    actor = StateFeedbackReg(env)
    
    state = env.reset()
    #env.pvtol.state = np.array([0, 0, np.deg2rad(10), 0, 0, 0])
    #env.render()
    #time.sleep(10)
    
    for i in range(1000):
        env.render()
        action =  actor.predict(state)
        state, reward, done, info = env.step(action)
        if done: break
        
