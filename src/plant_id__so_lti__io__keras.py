#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras
import utils as ut, so_lti

import pdb


'''
learning converges once in a while.... wtf!!!
'''


LOG = logging.getLogger('so_lti_plant_id_ff_sklearn')

class ANN_Plant:
    delay = 2
    x1_km1, x1_km2, u_km1, u_km2, input_size = range(5)

    def __init__(self, use_sequential_interface=False):
        if use_sequential_interface: # sequential interface
            self.ann = keras.models.Sequential()
            self.ann.add(keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_dim=4, use_bias=False))
        else: # functional interface
            a = keras.layers.Input(shape=(4,))
            b = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(4,), use_bias=False)(a)
            self.ann = keras.models.Model(inputs=a, outputs=b)
        self.ann.compile(loss='mean_squared_error', optimizer='adam')
        self.ann.summary()
        
    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x1_km1] = X[i-1, 0]
            _input[i-self.delay, self.x1_km2] = X[i-2, 0]
            _input[i-self.delay, self.u_km1] = U[i-1]
            _input[i-self.delay, self.u_km2] = U[i-2]
        return _input

    def fit(self, time, X, U, epochs):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:,0]
        print('fiting set')
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        history = self.ann.fit(ann_input, ann_output, epochs=epochs, batch_size=128,
                               verbose=1, shuffle=True, validation_split=0.2, callbacks=[early_stopping])
        print(' done')
        #print('score: {:e}'.format(self.ann.score(ann_input , ann_output)))
        print self.ann.get_layer(name='dense_1').get_weights()
        
    def get(self, x1_km1, x1_km2, u_km1, u_km2):
        return self.ann.predict(np.array([[x1_km1, x1_km2, u_km1, u_km2]]))

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
        LOG.info('{}'.format(w))

    def force_weights(self, plant):
        print "current weights"
        self.summary()
        self.ann.coefs_[0] = np.array([[-plant.b1], [-plant.b0], [plant.a1], [plant.a0]])
        self.ann.intercepts_[0] = np.array([ 0.])
        print "forced weights"
        self.summary()
        pdb.set_trace()

        
    
def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/so_lti__plant_id__io__keras.h5'

    omega, xi, dt= 3., 0.2, 0.01
    plant, ctl = so_lti.CCPlant(omega, xi, dt), ut.CtlNone()
    plant.analyse()
    ann = ANN_Plant()
    if train:
        time, X, U, desc = so_lti.make_or_load_training_set(plant, ctl, make_training_set, nsamples=int(100*1e3))
        ann.fit(time, X, U, epochs=100)
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
        plt.savefig('../docs/plots/plant_id__so_lti__io__keras.png')
        plt.show()
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=False, train=True)
