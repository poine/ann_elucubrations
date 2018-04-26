#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle
import keras, recurrentshop

import lti_first_order as fo, utils as ut
import pdb
LOG = logging.getLogger('mrc__fo_lti')

'''
model reference control on a first order LTI plant

# truth: [[ 100.5008316 ], [ -82.2831192 ], [ -17.21771049]]
# forced plant, 1000 epochs, 16 batch size: loss 5.5550e-05 weights  [[ 100.22059631], [ -69.12407684], [ -30.09073448]]
# trained plant, 1000 epochs, 16 batch size: loss 6.0758e-05 weights [[ 100.18427277], [ -68.48072815], [ -30.69457626]]
#
#

'''

class Controller:
    def __init__(self, dt, tau_ref, tau_err):
        self.dt, self.tau_err, self.tau_ref = dt, tau_err, tau_ref
        self.ref = fo.Plant(tau_ref, dt)
        self.Xr = 0

        ctl_i = keras.layers.Input((3,), name ="ctl_i") # Xr_kp1, Xr_k, X_k
        self.ctl_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(3,), use_bias=False, name="ctl")
        ctl_o = self.ctl_l(ctl_i)
        LOG.info(' ctl:\n{}'.format(self.ctl_l.get_weights()))
        self.ann_ctl = keras.models.Model(inputs=ctl_i, outputs=ctl_o)
        self.ann_ctl.summary()
        #self.ann_ctl.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
    def get(self, k, Xk):
        Xr_kp1 = self.ref.disc_dyn2(self.Xr, self.sp[k])
        U = self.ann_ctl.predict(np.array([[Xr_kp1, self.Xr, Xk]]))
        self.Xr = Xr_kp1
        #pdb.set_trace()
        return self.Xr, U

    def set_ctl_weights(self, w_xr_kp1, w_xr_k, w_x_k):
        ctl_l = self.ann_ctl.get_layer(name="ctl")
        ctl_l.set_weights([np.array([[w_xr_kp1], [w_xr_k], [w_x_k]])])
        LOG.info(' ctl:\n{}'.format(ctl_l.get_weights()))


    def force_ctl_truth(self, plant):
        a_e = np.exp(-plant.dt/self.tau_err)
        w_xr_kp1 = 1./plant.bd
        w_xr_k = -a_e/plant.bd
        w_x_k = (a_e-plant.ad)/plant.bd
        self.set_ctl_weights(w_xr_kp1, w_xr_k, w_x_k)

    def force_plant_truth(self, plant):
        self.plant_l.set_weights([np.array([[plant.bd],[plant.ad]])])
        LOG.info('Forcing plant ann weights\n{}'.format(self.plant_l.get_weights()))
 
    def train_plant(self, plant, train=True, make_training_set=True):
        filename = "/tmp/foo.h5"
        if train:
            plant_i = keras.layers.Input((2,), name ="plant_i") # u_k, x_k
            self.plant_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_shape=(2,), use_bias=False, name="plant")
            plant_o = self.plant_l(plant_i)
            plant_ann = keras.models.Model(inputs=plant_i, outputs=plant_o)
            plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            time, X, U, desc = fo.make_or_load_training_set(plant, ut.CtlNone(), make_training_set)

            plant_input, plant_output = np.hstack((U[:-1], X[:-1])), X[1:]
            plant_ann.fit(plant_input, plant_output, epochs=30, batch_size=32,  verbose=0, shuffle=True)
            plant_ann.save(filename)
        else:
            plant_ann = keras.models.load_model(filename)
            self.plant_l = plant_ann.get_layer(name="plant")
        
        LOG.info(' trained plant:\n{}'.format(self.plant_l.get_weights()))
    

    def train_control(self, train, epochs=1000):
        filename = "/tmp/mrc__fo_lti__ctl.h5"
        if train:
            ref_input = keras.layers.Input((2,), name ="ctl_i")   # Xr_kp1, Xr_k         
            state_input = keras.layers.Input((1,), name ="x_i")   # X_k
            ctl_input = keras.layers.concatenate([ref_input, state_input])
            ctl_output = self.ctl_l(ctl_input)

            plant_input =  keras.layers.concatenate([ctl_output, state_input]) # U_k, X_k
            self.plant_l.trainable = False
            plant_output = self.plant_l(plant_input)

            full_ann = keras.models.Model(inputs=[ref_input, state_input], outputs=plant_output)
            full_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            full_ann.summary()
    
            time, Xr, Ur, desc = fo.make_or_load_training_set(self.ref, ut.CtlNone(), True)
            _len = len(Xr)-1
            eps_k = np.random.uniform(low=-0.1, high=0.1, size=_len) # tracking error
            a_eps = np.exp(-self.dt/self.tau_err)
            eps_kp1 = a_eps*eps_k
            _input = [np.zeros((_len, 2)), np.zeros((_len, 1))] 
            _output = np.zeros((_len))
            for k in range(_len):
                _input[0][k] = [Xr[k+1], Xr[k]]
                _input[1][k] = [Xr[k]] + eps_k[k]
                _output[k] = Xr[k+1] + eps_kp1[k]
        
            full_ann.fit(_input, _output, epochs=epochs, batch_size=16, verbose=1, shuffle=True)
            full_ann.save(filename)
        else:
            full_ann = keras.models.load_model(filename)
            self.ctl_l = full_ann.get_layer(name="ctl")
            ctl_i = keras.layers.Input((3,), name ="ctl_i") # Xr_kp1, Xr_k, X_k
            self.ann_ctl = keras.models.Model(inputs=ctl_i, outputs=self.ctl_l(ctl_i))
        LOG.info(' ctl:\n{}'.format( self.ctl_l.get_weights()))
    

def test_control(plant, ctl):
    dt = plant.dt
    time =  np.arange(0., 8.05, dt)
    sp, Xr = ut.step_vec(time, dt=8), np.zeros((len(time),1))
    X, U = np.zeros((len(time),1)), np.zeros((len(time),1))
    ctl.sp = sp
    X[0] = 1.2
    for k in range(0,len(time)-1):
        Xr[k+1], U[k] = ctl.get(k, X[k])
        X[k+1] = plant.disc_dyn2(X[k], U[k])
        
    _unused, U[-1] = ctl.get(-1, X[-1])

    fo.plot(time, X, U, Xr)
    plt.savefig('../docs/plots/mrc__fo_lti_test1.png')
    plt.show()

def main():
    dt = 0.01
    plant = fo.Plant(1., dt)

    tau_ref, tau_err = 0.2, 0.05
    ctl = Controller(dt, tau_ref, tau_err)

    ctl.train_plant(plant, train=False, make_training_set=True)
    #ctl.force_plant_truth(plant)

    ctl.train_control(train=True, epochs=1000)

     
    test_control(plant, ctl)

    ctl.force_ctl_truth(plant)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()

