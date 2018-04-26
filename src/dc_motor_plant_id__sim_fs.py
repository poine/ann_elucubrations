#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, pickle, scipy.signal
import sklearn.neural_network
import utils, dc_motor

import pdb
LOG = logging.getLogger('dc_motor_plant_id_sim_fs')

class ANN_Plant:
    delay = 1
    th_km1, thd_km1, i_km1, v_km1, input_size = range(5)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(),     # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':False, 
            'random_state':1, 
            'max_iter':500, 
            'tol':1e-20,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)
        self.scale_input = False

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.th_km1] = X[i-1, dc_motor.Plant.s_phi]
            _input[i-self.delay, self.thd_km1] = X[i-1, dc_motor.Plant.s_om]
            _input[i-self.delay, self.i_km1] = X[i-1, dc_motor.Plant.s_ia]
            _input[i-self.delay, self.v_km1] = U[i-1, dc_motor.Plant.i_va]
        return _input

    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:]
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
        c, w = self.ann.coefs_[0], self.ann.intercepts_[0]
        #pdb.set_trace()
        # TODO - fix when scaler is used
        self.Ad = c[:3].T
        self.Bd = c[3].reshape((3,1))
        LOG.info('\nAd\n{}\nBd\n{}'.format(self.Ad, self.Bd))

        
    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), dc_motor.Plant.s_size)),  np.zeros((len(time), dc_motor.Plant.i_size))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            _input = self.scaler.transform([[X[i-1,0], X[i-1,1], X[i-1,2], U[i-1,0]]]) if self.scale_input else [[X[i-1,0], X[i-1,1], X[i-1,2], U[i-1,0]]]
            X[i] = self.ann.predict(_input)
        U[-1] = U[-2]
        return X, U       

def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/dc_motor_plant_fs_ann.pkl'

    plant = dc_motor.Plant()
    ann = ANN_Plant()
    if train:
        time, X, U, desc = dc_motor.make_or_load_training_set(plant, make_training_set)
        ann.fit(time, X, U)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)
        
    ann.summary()
    if test:
        time =  np.arange(0., 15.05, plant.dt)
        yc = utils.step_vec(time, dt=8)
        def ctl(X,k): return [yc[k], 0]
        X0 = [0, 0, 0]
        X1, U1 = plant.sim(time, X0, ctl)
        X2, U2 = ann.sim(time, X0, ctl)
        dc_motor.plot(time, X1, U1)
        dc_motor.plot(time, X2, U2)
        plt.suptitle('test trajectory');plt.subplot(4,1,1); plt.legend(['plant','ANN'])
        plt.savefig('../docs/plots/plant_id__dc_motor__sim_fs.png')
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=True, train=True, test=True)
