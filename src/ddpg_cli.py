#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" 
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami , Antoine Drouin
"""

import os, shutil, logging, yaml
import numpy as np, pickle
import tensorflow as tf, tflearn
import gym
from gym import wrappers
import argparse, pprint as pp

import ddpg_utils, ddpg_agent, gym_foo

import pdb

LOG = logging.getLogger('ddpg_cli')

def main(config, args):
    config.dump()
    with ddpg_agent.Model(config) as model:
        if args['load']:
            model.load_agent(args['load_dir'])
        if args['train']:
            model.train_agent()
        if args['test']:
            model.test_agent()
        if args['save']:
            model.save_agent(args['save_dir'])
            


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = ddpg_agent.Config()
    parser = config.setup_cmd_line_parser()
    
    parser.add_argument('--train', help='train the agent', action='store_true')
    parser.add_argument('--test', help='test the trained agent', action='store_true')
    parser.add_argument('--save', help='save the trained agent', action='store_true')
    parser.add_argument('--save_dir', help='directory to save the agent', default='/tmp')
    parser.add_argument('--load', help='load the agent', action='store_true')
    parser.add_argument('--load_dir', help='directory from which to load the agent', default='/tmp')
     
    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    
    args = config.parse_cmd_line(parser)
    

    main(config, args)
