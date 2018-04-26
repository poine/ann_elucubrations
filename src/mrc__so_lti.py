#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''

Model Reference Control of a second order, single input, linear, time invariant plant.

'''


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, scipy
import keras, control
import utils as ut, so_lti

import pdb
LOG = logging.getLogger('mrc__so_lti')


class Controller:
    def __init__(self, dt, omega_ref, xi_ref, omega_err, xi_err):
        self.ref = so_lti.CCPlant(omega_ref, xi_ref)
        self.Xr = np.zeros((2,1))

        self.track_err = so_lti.CCPlant(omega_err, xi_err)

        ctl_i = keras.layers.Input((6,), name ="ctl_i") # Xr1_kp1, Xr2_kp1, Xr1_k, Xr2_k , X1_k, X2_k
        self.ctl_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(6,), use_bias=False, name="ctl")
        ctl_o = self.ctl_l(ctl_i)
        LOG.info(' ctl:\n{}'.format(self.ctl_l.get_weights()))
        self.ann_ctl = keras.models.Model(inputs=ctl_i, outputs=ctl_o)
        self.ann_ctl.summary()

    def train_plant(self, plant, train=True, make_training_set=True):
        filename = "/tmp/mrc__so_lti__plant.h5" 
        if train:
            plant_i = keras.layers.Input((3,), name ="plant_i") # u_k, x1_k, x2_k
            self.plant_l = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform', input_shape=(3,), use_bias=False, name="plant")
            plant_o = self.plant_l(plant_i)
            plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
            plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            time, X, U, desc = so_lti.make_or_load_training_set(plant, ut.CtlNone(), make_training_set)
            _input = np.vstack([U[:-1,0], X[:-1,0], X[:-1,1]]).T
            _output = np.vstack([X[1:,0], X[1:,1]]).T
            plant_ann.fit(_input, _output, epochs=20, batch_size=32,  verbose=1, shuffle=True)
            plant_ann.save(filename)
        else:
            plant_ann = keras.models.load_model(filename)
            self.plant_l = plant_ann.get_layer(name="plant")

        LOG.info(' trained plant:\n{}'.format(self.plant_l.get_weights()))

    def train_control(self, train=True, epochs=1000):
        filename = "/tmp/mrc__so_lti__ctl.h5"
        if train:
            ref_input = keras.layers.Input((4,), name ="ctl_i")   # Xr1_kp1, Xr2_kp1, Xr1_k, Xr2_k         
            state_input = keras.layers.Input((2,), name ="x_i")   # X1_k, X2_k
            ctl_input = keras.layers.concatenate([ref_input, state_input])
            ctl_output = self.ctl_l(ctl_input)
        
            plant_input =  keras.layers.concatenate([ctl_output, state_input]) # U_k, X1_k, X2_k
            self.plant_l.trainable = False
            plant_output = self.plant_l(plant_input)
        
            full_ann = keras.models.Model(inputs=[ref_input, state_input], outputs=plant_output)
            full_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            full_ann.summary()

        
            time, Xr, Ur, desc = so_lti.make_or_load_training_set(self.ref, ut.CtlNone(), True)
            _len = len(Xr)-1
            eps_k = np.random.uniform(low=-1., high=1., size=(_len,2)) # tracking error
            eps_kp1 = np.array([np.dot(self.track_err.Ad, _eps_k) for _eps_k in eps_k])

            _input = [np.zeros((_len, 4)), np.zeros((_len, 2))] 
            _output = np.zeros((_len, 2))
            for k in range(_len):
                _input[0][k] = [Xr[k+1, 0], Xr[k+1, 1], Xr[k, 0], Xr[k, 1]]
                _input[1][k] = Xr[k] + eps_k[k]
                _output[k] = Xr[k+1] + eps_kp1[k]
        
            full_ann.fit(_input, _output, epochs=epochs, batch_size=16, verbose=1, shuffle=True)
            full_ann.save(filename)
        else:
            full_ann = keras.models.load_model(filename)
            self.ctl_l = full_ann.get_layer(name="ctl")
            ctl_i = keras.layers.Input((6,), name ="ctl_i") # Xr1_kp1, Xr2_kp1, Xr1_k, Xr2_k , X1_k, X2_k
            ctl_o = self.ctl_l(ctl_i)
            self.ann_ctl = keras.models.Model(inputs=ctl_i, outputs=ctl_o)
            
        LOG.info(' ctl:\n{}'.format( self.ctl_l.get_weights()))
    

        
    def get(self, k, Xk):
        Xr_kp1 = self.ref.disc_dyn(self.Xr, self.sp[k])
        U = self.ann_ctl.predict(np.array([[Xr_kp1[0], Xr_kp1[1], self.Xr[0], self.Xr[1], Xk[0], Xk[1]]]))
        self.Xr = Xr_kp1
        return self.Xr.squeeze(), U.squeeze()


def test_control(plant, ctl):
    time =  np.arange(0., 8.05, plant.dt)
    sp, Xr = ut.step_vec(time, dt=8), np.zeros((len(time),2))
    X, U = np.zeros((len(time),2)), np.zeros((len(time),1))
    ctl.sp = sp
    X[0] = [1.2, 0]
    for k in range(0,len(time)-1):
        #pdb.set_trace()
        Xr[k+1], U[k] = ctl.get(k, X[k])
        X[k+1] = plant.disc_dyn(X[k], U[k])
        
    _unused, U[-1] = ctl.get(-1, X[-1])

    so_lti.plot(time, X, U, Xr)
    plt.savefig('../docs/plots/mrc__so_lti_test1.png')
    plt.show()


from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

def main(train_plant=False, train_ctl=False):
    print K.tensorflow_backend._get_available_gpus()
    print(device_lib.list_local_devices())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    dt = 0.01
    plant = so_lti.CCPlant(omega=1, xi=0.5, dt=dt)
    
    omega_ref, xi_ref, omega_err, xi_err = 3., 1., 10., 0.7
    ctl = Controller(dt, omega_ref, xi_ref, omega_err, xi_err)

    ctl.train_plant(plant, train=train_plant, make_training_set=False)

    ctl.train_control(train=train_ctl, epochs=1000)

    test_control(plant, ctl)
    
    return
  
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
