#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import sklearn.neural_network
import utils as ut, so_lti

import pdb


'''
learning converges once in a while.... wtf!!!
'''


LOG = logging.getLogger('so_lti_plant_id_ff_sklearn')

class ANN_Plant:
    delay = 2
    x1_km1, x1_km2, u_km1, u_km2, input_size = range(5)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(),     # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':False, 
            'random_state':1, 
            'max_iter':500, 
            'tol':1e-16,
            'warm_start': False
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x1_km1] = X[i-1, 0]
            _input[i-self.delay, self.x1_km2] = X[i-2, 0]
            _input[i-self.delay, self.u_km1] = U[i-1]
            _input[i-self.delay, self.u_km2] = U[i-2]
        return _input

    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:,0]
        #print(' done, now scaling training set')
        #self.scaler = sklearn.preprocessing.StandardScaler()
        #self.scaler.fit(ann_input)
        #scaled_input = self.scaler.transform(ann_input)
        #print('fiting set')
        self.scaler=None
        self.ann.fit(ann_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(ann_input , ann_output)))

    def get(self, x1_km1, x1_km2, u_km1, u_km2):
        #return self.ann.predict(self.scaler.transform([[x1_km1, x2_km1, ukm1]]))
        return self.ann.predict([[x1_km1, x1_km2, u_km1, u_km2]])

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 2)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i,0] = self.get(X[i-1,0], X[i-2,0], U[i-1,0], U[i-2,0])
        U[-1] = U[-2]
        X[:,1] = float('nan')
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
        c, w = self.ann.coefs_, self.ann.intercepts_
        #Ad = np.hstack((c[0][0].reshape(2,1), c[0][1].reshape(2,1)))
        #Bd = c[0][2].reshape(2,1)
        #LOG.info(' recovered SSR\nAd\n{}\nBd\n{}'.format(Ad, Bd))
        LOG.info('{} {}'.format(c, w))

    def force_weights(self, plant):
        print "current weights"
        self.summary()
        self.ann.coefs_[0] = np.array([[-plant.b1], [-plant.b0], [plant.a1], [plant.a0]])
        self.ann.intercepts_[0] = np.array([ 0.])
        print "forced weights"
        self.summary()
        pdb.set_trace()

        
    
def main(make_training_set=False, train=True, test=True):
    ann_plant_filename = '/tmp/so_lti__plant_id__io__sklearn.pkl'

    omega, xi, dt= 3., 0.2, 0.01
    plant, ctl = so_lti.CCPlant(omega, xi, dt), ut.CtlNone()
    plant.analyse()
    ann = ANN_Plant()
    if train:
        time, X, U, desc = so_lti.make_or_load_training_set(plant, ctl, make_training_set, nsamples=int(100*1e3))
        ann.fit(time, X, U)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)

    ann.summary()
    #ann.force_weights(plant)

    if test:
        time =  np.arange(0., 15.05, plant.dt)
        ctl.yc = ut.step_vec(time, dt=8)
        X0 = [0]
        X1, U1 = plant.sim(time, X0, ctl.get)
        #X1_, U1_ = plant.sim_io(time, X0, ctl.get)
        X2, U2 = ann.sim(time, X0, ctl.get)
        so_lti.plot(time, X1, U1)
        #so_lti.plot(time, X1_, U1_)
        so_lti.plot(time, X2, U2)
        plt.suptitle('test trajectory');plt.subplot(3,1,1); plt.legend(['plant','ANN'])
        plt.savefig('../docs/plots/plant_id__so_lti__io__sklearn.png')
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=False, train=True)
