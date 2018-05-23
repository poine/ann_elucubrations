#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, gym
import dql_utils, gym_foo

if __name__ == '__main__':
    env_name = 'planar_quad-v0'
    env = gym.make(env_name)
    env.seed(123)
    s = env.reset()
    for i in range(1000):
        env.render()
        action = np.array([0.25*9.8, 0.25*9.8])
        state, reward, done, info = env.step(action)
        if done: break
        
