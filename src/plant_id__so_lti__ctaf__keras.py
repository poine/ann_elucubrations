#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras
import utils as ut, so_lti

import pdb


'''
Plant Identification under Control Affine Form
'''

LOG = logging.getLogger('plant_id__so_lti__ctaf__keras')

class ANN_Plant:
    delay = 2
    x1_km1, x1_km2, u_km1, u_km2, input_size = range(5)

    def __init__(self):
        if 1:
            _i1 = keras.layers.Input(shape=(3,)) # x_km1, x_km2, u_km2
            _i2 = keras.layers.Input(shape=(1,)) # u_km1
            _tf = keras.layers.Dense(1, input_shape=(3,), use_bias=False, name='f')(_i1)
            _tg = keras.layers.Dense(1, input_shape=(3,), use_bias=True, name='g')(_i1)
            _t_ug = keras.layers.multiply([_i2, _tg])
            _t_xkp1 = keras.layers.add([_tf, _t_ug])
            self.ann = keras.models.Model(inputs=[_i1, _i2], output=_t_xkp1)
        else:
            a = keras.layers.Input(shape=(4,))
            b = keras.layers.Dense(1, input_shape=(4,), use_bias=False)(a)
            self.ann = keras.models.Model(inputs=a, outputs=b)
        self.ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.ann.summary()

    def make_input(self, X, U):
        _i1 = np.zeros(( len(X)- self.delay, 3))
        _i2 = np.zeros(( len(X)- self.delay, 1))
        for i in range(self.delay, len(X)):
            _i1[i-self.delay, 0] = X[i-1, 0]
            _i1[i-self.delay, 1] = X[i-2, 0]
            _i1[i-self.delay, 2] = U[i-2]
            _i2[i-self.delay, 0] = U[i-1]
        return _i1, _i2

    def fit(self, time, X, U):  
        print('building training set')
        (ann_input1, ann_input2), ann_output = self.make_input(X, U), X[self.delay:,0]
        print('fiting set')
        self.ann.fit([ann_input1, ann_input2], ann_output, epochs=20, batch_size=32,  verbose=1, shuffle=True)
        print(' done')
        #print('score: {:e}'.format(self.ann.score(ann_input , ann_output)))

    def get(self, x1_km1, x1_km2, u_km1, u_km2):
        return self.ann.predict([np.array([[x1_km1, x1_km2, u_km2]]), np.array([[u_km1]])])

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
        self.ann.save(filename+'.h5')


    def load(self, filename):
        LOG.info(' Loading ann from {}'.format(filename))
        self.ann = keras.models.load_model(filename+'.h5')

    def summary(self):
        w = self.ann.layers[0].get_weights()
        #Ad = np.hstack((c[0][0].reshape(2,1), c[0][1].reshape(2,1)))
        #Bd = c[0][2].reshape(2,1)
        #LOG.info(' recovered SSR\nAd\n{}\nBd\n{}'.format(Ad, Bd))
        LOG.info('f: {}'.format(self.ann.get_layer(name='f').get_weights()))
        LOG.info('g: {}'.format(self.ann.get_layer(name='g').get_weights()))


        
    
def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/so_lti__plant_id__ctaf__sklearn.pkl'

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
        X2, U2 = ann.sim(time, X0, ctl.get)
        so_lti.plot(time, X1, U1)
        so_lti.plot(time, X2, U2)
        plt.suptitle('test trajectory');plt.subplot(3,1,1); plt.legend(['plant','ANN'])
        plt.savefig('../docs/plots/plant_id__so_lti__ctaf__keras.png')
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=False, train=False)
