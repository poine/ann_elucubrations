#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import sklearn.neural_network
import utils as ut, lti_first_order as ltifo

import pdb

LOG = logging.getLogger('fo_lti_id_plant_feedforward_sklearn')

class ANN_Plant:
    ann_delay = 1
    y_km1, u_km1, ann_input_size = range(3)
    
    def __init__(self):
        params = {
            'hidden_layer_sizes': (),    # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose': False, 
            'random_state': 1, 
            'max_iter': 500, 
            'tol': 1e-12,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)
        self.scaler = None
        
    def fit(self, time, X, U):  
        LOG.info(' Fitting Plant ANN on {} data points'.format(len(time)))
        LOG.info('  preparing training set')
        n_samples = len(time)-self.ann_delay
        ann_input, ann_output = np.zeros((n_samples, self.ann_input_size)), np.zeros(n_samples)
        for i in range(self.ann_delay, len(time)):
            ann_output[i-self.ann_delay] = X[i]
            ann_input[i-self.ann_delay, self.y_km1] = X[i-1]
            ann_input[i-self.ann_delay, self.u_km1] = U[i-1]
        LOG.info('  done. Now fitting set')
        self.ann.fit(ann_input, ann_output)
        LOG.info('  done')
        LOG.info('  score: {:f}'.format(self.ann.score(ann_input , ann_output)))

    def get(self, Xkm1, Ukm1):
        return self.ann.predict([[Xkm1, Ukm1]])
    
    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 1)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.get(X[i-1,0], U[i-1,0])
        U[-1] = U[-2]
        return X, U

    def save(self, filename):
        LOG.info('  saving ann to {}'.format(filename))
        with open(filename, "wb") as f:
            pickle.dump([self.ann, self.scaler], f)

    def load(self, filename):
        LOG.info(' Loading ann from {}'.format(filename))
        with open(filename, "rb") as f:
            self.ann, self.scaler = pickle.load(f)

    def summary(self):
        (a, b), c = self.ann.coefs_[0], self.ann.intercepts_[0]
        LOG.info(' xkp1 = {:.5f} xk + {:.5f} uk + {:.5f}'.format(a[0], b[0], c[0]))
        


def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/frst_order_plant_ann.pkl'

    tau, dt = 1., 1./100
    plant = ltifo.Plant(tau, dt)
    ctl = ut.CtlNone()
    ann = ANN_Plant()
    if train:
        time, X, U, desc =  ltifo.make_or_load_training_set(plant, ctl, make_training_set)
        ann.fit(time, X, U)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)

    ann.summary()
    if test:
        time =  np.arange(0., 15.05, plant.dt)
        ctl.yc = ut.step_input_vec(time, dt=8)
        X0 = [0]
        X1, U1 = plant.sim(time, X0, ctl.get)
        X2, U2 = ann.sim(time, X0, ctl.get)
        ltifo.plot(time, X1, U1)
        ltifo.plot(time, X2, U2)
        plt.legend(['plant','ANN'])
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=True, train=True)

    
