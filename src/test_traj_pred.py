#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, shutil, logging, pdb, matplotlib.pyplot as plt
import numpy as np, pickle, itertools
import utils as ut

"""
Unit convertions
"""
# http://en.wikipedia.org/wiki/Nautical_mile
def m_of_NM(nm): return nm*1852.
def NM_of_m(m): return m/1852.
# http://en.wikipedia.org/wiki/Knot_(speed)
def mps_of_kt(kt): return kt*0.514444
def kt_of_mps(mps): return mps/0.514444

'''
the rest
'''

def time_sec_of_str(s):
    toks = s.split(':')
    return int(toks[2]) + 60*int(toks[1]) + 60*60*int(toks[0])

class Trajectory:
    def __init__(self, _id):
        self._id = _id
        self._tmp = []
        
    def append(self, t, x, y, vx, vy):
        self._tmp.append([t, x, y, vx, vy])

    def finalyze(self):
        self.points = np.array(self._tmp)
        self.time = self.points[:,0]
        self.pos = self.points[:,1:3]
        self.vel = self.points[:,3:5]
        del self._tmp
    
class DataSet:
    def __init__(self, **kwargs):
        if 'parse' in kwargs: self.parse(kwargs['parse'])
        if 'load' in kwargs: self.load(kwargs['load'])


    def parse(self, filename):
        print('parsing trajectorie from {}'.format(filename))
        self.trajectories = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('Version:') or line.startswith('Centre:') or line.startswith('NbVols:') or line.startswith('NbPlots:'): continue
                elif line.startswith('$'):
                    # $ FLIGHT HDEB HFIN FL SPEED IVOL AV TERD TERA SSR RVSM TCAS ADSB
                    toks = line.split()
                    flight_id = toks[1]
                    self.trajectories.append(Trajectory([flight_id]))
                    #print 'pln', toks
                elif line.startswith('!') or line.startswith('>') or line.startswith('<') or line.startswith('%'):
                    pass
                    #print 'blah'
                else:
                    toks = line.split()
                    # HEURE X(1/64 Nm) Y(1/64 Nm) VX(Kts) VY(Kts) FL TAUX(Ft/min) TENDANCE
                    try:
                        t, x, y = time_sec_of_str(toks[0]), m_of_NM(float(toks[1])/64), m_of_NM(float(toks[2])/64)
                        vx, vy, fl = mps_of_kt(float(toks[3])), mps_of_kt(float(toks[4])), float(toks[5])
                        #taux, tend = float(mps_of_kt(toks[6])), toks[7]
                        #pdb.set_trace()
                        #print toks
                        self.trajectories[-1].append(t, x, y, vx, vy)
                    except IndexError:
                        #print toks
                        pass
        for t in self.trajectories: t.finalyze()

        print('  found {} trajectories'.format(len(self.trajectories)))
        

    def save(self, filename):
         print('Saving trajectorie to {}'.format(filename))
         with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f)

    def load(self, filename):
        print('Loading trajectorie from {}'.format(filename))
        with open(filename, 'rb') as f:
            self.trajectories = pickle.load(f)
        print('  found {} trajectories'.format(len(self.trajectories)))
        
    def get(self):
        pass



class Agent_0:
    ''' first order prediction using positions and velocity (zero acceleration) '''
    def __init__(self):
        self.delay = 0

    def predict(self, traj, k, horiz):
        return traj.pos[k] + traj.vel[k]*horiz


class Agent_1:
    ''' first order prediction using positions only (numerical differentiation, zero acceleration)'''
    def __init__(self):
        self.delay = 1

    def predict(self, traj, k, horiz):
        return traj.pos[k] + (traj.pos[k]-traj.pos[k-1])*horiz


class Agent_2:
    ''' second order prediction (constant acceleration)'''
    def __init__(self):
        self.delay = 1

    def predict(self, traj, k, horiz):
        Ak = traj.vel[k] - traj.vel[k-1]
        return traj.pos[k] + traj.vel[k]*horiz + Ak*(horiz**2)/.2

class Agent_ann0:
    def __init__(self, delay=1):
        self.delay = delay

    def predict(self, traj, k, horiz):
        return traj.pos[k]
        
    
def plot(ds, which=None):
    if which is None:
        ut.prepare_fig(fig=None, window_title=None, figsize=(10.24, 10.24), margins=None)
        for t in ds.trajectories:
            plt.plot(t.points[:,1], t.points[:,2])
        plt.axes().set_aspect('equal')
        ut.decorate(plt.gca(), title='Two D view ({} trajectories)'.format(len(ds.trajectories)), xlab='East (m)', ylab='North (m)', legend=None, xlim=None, ylim=None)
        ut.save_if('../docs/plots/traj_pred_2D.png')
        plt.figure()
        for t in ds.trajectories:
            plt.plot(t.points[:,0], np.linalg.norm(t.points[:,3:5], axis=1))
    else:
        plt.figure()
        plt.plot(ds.trajectories[which].points[:,1], ds.trajectories[which].points[:,2])
        plt.figure()
        plt.plot(ds.trajectories[which].points[:,0], np.linalg.norm(ds.trajectories[which].points[:,3:5], axis=1))



def eval_prediction(agent, trajectory, horiz):
    p_errs = []
    for idx in range(agent.delay, len(trajectory.points)-horiz):
        p1 = agent.predict(trajectory, idx, float(horiz))
        p2 = trajectory.points[idx+horiz, 1:3]
        p_errs.append(np.linalg.norm(p2 - p1))
    return p_errs

def test_prediction(agent, trajectory, horiz):
    p_errs = eval_prediction(agent, trajectory, horiz)
    plt.figure()
    plt.plot(p_errs)
    plt.figure()
    plt.hist(p_errs, bins=100, density=True)


def test_predictions(agent, trajectories, horiz=[1]):
    margins=(0.04, 0.1, 0.98, 0.91, 0.31, 0.2)
    ut.prepare_fig(fig=None, window_title=None, figsize=(20.48, 5.12), margins=margins)
    for i, h in enumerate(horiz):
        p_errs = list(itertools.chain.from_iterable([eval_prediction(agent, traj, h) for traj in trajectories]))
        ax = plt.subplot(1, len(horiz), i+1) 
        plt.hist(p_errs, bins=100, normed=True)#density=True)
        mu = np.mean(p_errs)
        ut.decorate(ax, title='horiz: {} s $\mu$: {:.0f} m'.format(h, mu), xlab='error (m)', ylab='density', legend=None, xlim=[0, 750], ylim=None)
    #pdb.set_trace()
    


def convert_dataset(rjx_file='../data/bdx_20130914_25ft.txt', pkl_file='../data/bdx_20130914_25ft.pkl'):
    ''' parses rejeux ascii file and saves pickle object'''
    d = DataSet(parse=rjx_file)
    d.save(pkl_file)
    return d
    
def plot_dataset(ds):
    plot(ds)
    
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    convert_dataset()
    d = DataSet(load='../data/bdx_20130914_25ft.pkl')
    # plot_dataset(d); plt.show()

    # if 0:
    #     d.parse('../data/bdx_20130914_25ft.txt')
    #     d.save('../data/bdx_20130914_25ft.pkl')
    # else:
    #     d.load('../data/bdx_20130914_25ft.pkl')
    #     if 0:
    #         for i in range(len(d.trajectories)):
    #             print d.trajectories[i]._id
    #             plot(d, i)
    #             plt.show()
    #     if 0:
    #          plot(d)
    #          plt.show()

    #     a_types, horizons = [Agent_0, Agent_1, Agent_ann0], [1, 2, 5, 10]
    a_types, horizons = [Agent_2], [1, 5, 10, 20]
    for a_type in a_types:
        a = a_type()
        test_predictions(a, d.trajectories, horizons)
    ut.save_if('../docs/plots/traj_pred_first_order_num_diff.png')
    plt.show()
