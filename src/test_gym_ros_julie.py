#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, signal, os, time, logging, yaml, numpy as np, matplotlib.pyplot as plt
import gym, gym_foo
import two_d_guidance as tdg
import pdb

def signal_handler(signal, frame):
    print('sigint caught')
    sys.exit(0)

def test_1(env_name='julie-v0', env_cfg='ddpg_run_julie_01.yaml'):
    env = gym.make(env_name)
    with open(env_cfg, 'r') as stream:
            cfg = yaml.load(stream)
    env.load_config(cfg['env'])
    env.seed()
    time.sleep(1)
    param = type('Param', (), {})(); param.L = 1.65
    ctl =  tdg.PurePursuitVelController(env.path_filename, param, look_ahead=env.carrot_dists[0], v_sp=env.v_sp)
    env.reset([1, 0, 0.1])
    print('state {}'.format(env.X))
    print('track err {} heading_err {}'.format(env.err_tracking, env.err_heading))
    env.render()
    time.sleep(10)
    return
    for i in range(10000):
        #print('{}'.format(i))
        x, y, psi, v = env.X
        U = ctl.compute([x, y], psi, v)
        X, reward, over, info = env.step(U[1:])
    print('done')
        
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    test_1('julie-v0', 'ddpg_run_julie_01.yaml')
    #test_1('bicycle-v0', 'ddpg_run_bicycle_02.yaml')
