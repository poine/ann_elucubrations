#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, pickle, itertools

"""
Unit convertions
"""
# http://en.wikipedia.org/wiki/Nautical_mile
def m_of_NM(nm): return nm*1852.
def NM_of_m(m): return m/1852.
# http://en.wikipedia.org/wiki/Knot_(speed)
def mps_of_kt(kt): return kt*0.514444
def kt_of_mps(mps): return mps/0.514444

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


