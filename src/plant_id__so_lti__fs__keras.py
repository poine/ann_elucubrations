#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras
import utils as ut, so_lti

import pdb

LOG = logging.getLogger('so_lti_plant_id_ff_sklearn')

class ANN_Plant:
    delay = 1
    x1_km1, x2_km1, u_km1, input_size = range(4)

    def __init__(self, use_sequential_interface=False):
        if use_sequential_interface: # sequential interface
            self.ann = keras.models.Sequential()
            self.ann.add(keras.layers.Dense(2, activation='linear', kernel_initializer='uniform', input_dim=3, use_bias=False, name="f"))
        else: # functional interface
            a = keras.layers.Input(shape=(3,))
            b = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform', input_shape=(3,), use_bias=False, name="f")(a)
            self.ann = keras.models.Model(inputs=a, outputs=b)
        self.ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.ann.summary()

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x1_km1] = X[i-1, 0]
            _input[i-self.delay, self.x2_km1] = X[i-1, 1]
            _input[i-self.delay, self.u_km1] = U[i-1]
        return _input

    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:]
        print(' done, now fitting')
        self.ann.fit(ann_input, ann_output, epochs=40, batch_size=32,  verbose=1, shuffle=True)
        print(' done')
        #print('score: {:e}'.format(self.ann.score(ann_input , ann_output)))

    def get(self, x1_km1, x2_km1, ukm1):
        return self.ann.predict(np.array([[x1_km1, x2_km1, ukm1]]))

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 2)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.get(X[i-1,0], X[i-1,1], U[i-1,0])
        U[-1] = U[-2]
        return X, U
    
    def save(self, filename):
        LOG.info('  saving ann to {}'.format(filename))
        self.ann.save(filename)

    def load(self, filename):
        LOG.info(' Loading ann from {}'.format(filename))
        self.ann = keras.models.load_model(filename)

    def summary(self):
        w = self.ann.get_layer(name="f").get_weights()
        LOG.info('{}'.format(w))
        

    
def main(make_training_set=False, train=True, test=True):
    ann_plant_filename = '/tmp/so_lti__plant_id__fs__keras.h5'

    omega, xi, dt= 3., 0.2, 0.01
    plant, ctl = so_lti.CCPlant(omega, xi, dt), ut.CtlNone()
    ann = ANN_Plant()
    if train:
        time, X, U, desc = so_lti.make_or_load_training_set(plant, ctl, make_training_set)
        ann.fit(time, X, U)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)

    ann.summary()
    if test:
        time =  np.arange(0., 15.05, plant.dt)
        ctl.yc = ut.step_vec(time, dt=8)
        X0 = [0]
        X1, U1 = plant.sim(time, X0, ctl.get)
        X2, U2 = ann.sim(time, X0, ctl.get)
        so_lti.plot(time, X1, U1)
        so_lti.plot(time, X2, U2)
        plt.suptitle('test trajectory');plt.subplot(3,1,1); plt.legend(['plant','ANN'])
        plt.savefig('../docs/plots/plant_id__so_lti__fs__keras.png')
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=True, train=True, test=True)
