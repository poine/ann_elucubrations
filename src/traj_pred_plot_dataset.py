#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import logging, sys, itertools, numpy as np, matplotlib.pyplot as plt

import utils as ut, traj_pred_utils as tpu
from traj_pred_utils import Trajectory

import pdb

def plot(filename):
    ds = tpu.DataSet(load=filename)
    ut.prepare_fig(fig=None, window_title=None, figsize=(10.24, 10.24), margins=None)

    ax = plt.gca()#subplot(1,2,1)
    for t in ds.trajectories:
        plt.plot(t.pos[:,0], t.pos[:,1])
    plt.axes().set_aspect('equal')
    ut.decorate(ax, title='Two D view ({})'.format(filename), xlab='East (m)', ylab='North (m)', legend=None, xlim=None, ylim=None)

    try:
        curvatures = list(itertools.chain.from_iterable([t.curv for t in ds.trajectories]))
        #pdb.set_trace()
        ut.prepare_fig(fig=None, window_title=None, figsize=(10.24, 10.24), margins=None)
        ax = plt.subplot(2,1,1)
        plt.hist(curvatures, density=True, bins=100)
        ut.decorate(ax, title='curvature')
    except AttributeError:
        print('no curvature, skipping')
    

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    filename = '../data/bdx_20130914_25ft.pkl' if len(sys.argv) <= 1 else sys.argv[1]
    plot(filename)
    plt.show()
