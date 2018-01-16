#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, pickle, scipy.signal
import sklearn.neural_network
import utils, dc_motor

import pdb
LOG = logging.getLogger('dc_motor_plant_id_sim_io')

class ANN_Plant:
    delay = 3
    th_km1, th_km2, th_km3, v_km1, v_km2, v_km3, input_size = range(7)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(),     # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose': False, 
            'random_state':1, 
            'max_iter':500, 
            'tol':1e-20,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)
        self.scale_input = False

    def make_training_set(self, X, U):
        batch_len = len(X)- self.delay
        _input  = np.zeros(( batch_len, self.input_size))
        _output = np.zeros(batch_len)
        for k in range(batch_len):
            _input[k, self.th_km1] = X[k+self.delay-1, dc_motor.Plant.s_phi]
            _input[k, self.th_km2] = X[k+self.delay-2, dc_motor.Plant.s_phi]
            _input[k, self.th_km3] = X[k+self.delay-3, dc_motor.Plant.s_phi]
            _input[k, self.v_km1]  = U[k+self.delay-1, dc_motor.Plant.i_va]
            _input[k, self.v_km2]  = U[k+self.delay-2, dc_motor.Plant.i_va]
            _input[k, self.v_km3]  = U[k+self.delay-3, dc_motor.Plant.i_va]
            _output[k] = X[k+self.delay, dc_motor.Plant.s_phi]
        return _input, _output

    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_training_set(X, U)
        print(' done, now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(ann_input)
        scaled_input = self.scaler.transform(ann_input)
        print('fiting set')
        _input = scaled_input if self.scale_input else ann_input
        self.ann.fit(_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(_input , ann_output)))

    def save(self, filename):
        LOG.info('  saving ann to {}'.format(filename))
        with open(filename, "wb") as f:
            pickle.dump([self.ann, self.scaler], f)

    def load(self, filename):
        LOG.info(' Loading ann from {}'.format(filename))
        with open(filename, "rb") as f:
            self.ann, self.scaler = pickle.load(f)

    def summary(self):
        c, w = self.ann.coefs_, self.ann.intercepts_
        print c, w

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), dc_motor.Plant.s_size)),  np.zeros((len(time), dc_motor.Plant.i_size))
        for i in range(self.delay):
            X[i] = X0
        for i in range(self.delay, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            _inp = [[X[i-1,0], X[i-2,0], X[i-3,0], U[i-1, 0], U[i-2, 0], U[i-3, 0]]]
            if self.scale_input:
                _inp = self.scaler.transform(_inp)
            X[i,0] = self.ann.predict(_inp)

        X[:,1] = float('nan')
        X[:,2] = float('nan')
        U[-1] = U[-2]
        return X, U

    def force_truth(self, _as, _bs):
        #pdb.set_trace()
        LOG.info(' Forcing ann coeffs to thruth')
        self.ann.coefs_[0][:3,0] = _as
        self.ann.coefs_[0][3:,0] = _bs
        self.ann.intercepts_[0][0] = 0

        
        
import control
def test_tf(plant):
    #def expAtB(X, t): return np.dot(scipy.linalg.expm(plant.Ac)*t, plant.Bc)
    #_unused, Bd = scipy.integrate.odeint(expAtB, plant.Bd, [0, plant.dt])
    ct_sys = control.ss(plant.Ac, plant.Bc, [[1, 0, 0]], [[0,0]])
    #print ct_sys
    #eival, eivec = np.linalg.eig(plant.Ac)
    ct_tf = control.ss2tf(ct_sys)
    #print ct_tf
    
    dt_sys = control.sample_system(ct_sys, plant.dt, method='zoh') #  ‘matched’, ‘tustin’, ‘zoh’ 
    #print dt_sys
    #print plant.Ad
    dt_tf = control.ss2tf(dt_sys)
    print dt_tf
    b2, b1, b0 = dt_tf.num[0][0]
    a3, a2, a1, a0 = dt_tf.den[0][0] # a3 is 1
    #pdb.set_trace()
    #print b2, b1, b0
    def dif_eq(th_km1, th_km2, th_km3, u_km1, u_km2, u_km3):
        xk = -a2*th_km1 -a1*th_km2 -a0*th_km3 + b2*u_km1 + b1*u_km2 + b0*u_km3
        return xk
        
    time =  np.arange(0., 15.05, plant.dt)
    us = utils.step_input_vec(time, dt=8)
    ths = np.zeros(len(time))
    for k in range(3,len(time)):
        ths[k] =  dif_eq(ths[k-1], ths[k-2], ths[k-3], us[k-1], us[k-2], us[k-3])
    #pdb.set_trace()
    return ths, [-a2, -a1, -a0], [b2, b1, b0]
    
    
def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/dc_motor_plant_io_ann.pkl'

    plant = dc_motor.Plant()
    ths, _as, _bs = test_tf(plant)

    ann = ANN_Plant()
    if train:
        time, X, U, desc = dc_motor.make_or_load_training_set(plant, make_training_set)
        ann.fit(time, X, U)
        ann.summary()
        ann.force_truth(_as, _bs)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)
        
    ann.summary()
    if test:
        time =  np.arange(0., 15.05, plant.dt)
        yc = utils.step_input_vec(time, dt=8)
        def ctl(X,k): return [yc[k], 0]
        X0 = [0, 0, 0]
        X1, U1 = plant.sim(time, X0, ctl)
        X2, U2 = ann.sim(time, X0, ctl)
        dc_motor.plot(time, X1, U1)
        dc_motor.plot(time, X2, U2)
        plt.suptitle('test trajectory');plt.subplot(4,1,1); plt.legend(['plant','ANN'])
        plt.plot(time, ths)
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=True, train=True, test=True)
