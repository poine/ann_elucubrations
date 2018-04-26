#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle
import keras
import robot_arm, so_lti, utils as ut
import pdb
LOG = logging.getLogger('mrc__robot_arm')

'''
model reference control on robot arm
'''

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


    def train_plant(self, plant, train=True, make_training_set=True, epochs=50):
        filename = "/tmp/mrc__robot_arm__plant.h5" 
        if train:
            plant_i = keras.layers.Input((3,), name ="plant_i") # phi_k, phid_k, u_k
            self.plant_l = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform', use_bias=False, name="plant")
            self.plant_ann = keras.models.Model(inputs=plant_i, outputs=self.plant_l(plant_i))
            self.plant_ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

            time, X, U, desc = robot_arm.make_or_load_training_set(plant, make_training_set, '/tmp/mrc__robot_arm__plant_training_traj.npz', nsamples=int(100*1e3))
            #robot_arm.plot(time, X, U)
            #plt.show()
            _input = np.vstack([X[:-1,0], X[:-1,1], U[:-1,0]]).T
            _output = np.vstack([X[1:,0], X[1:,1]]).T
            history = self.plant_ann.fit(_input, _output, epochs=epochs, batch_size=64,  verbose=1, shuffle=True)
            self.plant_ann.save(filename)
        else:
            self.plant_ann = keras.models.load_model(filename)
            self.plant_l = self.plant_ann.get_layer(name="plant")

        LOG.info(' trained plant:\n{}'.format(self.plant_l.get_weights()))


    def train_control(self, train=True, epochs=1000):
        filename = "/tmp/mrc__robot_arm__ctl.h5"
        if train:
            ref_input = keras.layers.Input((4,), name ="ctl_i")   # Xr1_kp1, Xr2_kp1, Xr1_k, Xr2_k         
            state_input = keras.layers.Input((2,), name ="x_i")   # X1_k, X2_k
            ctl_input = keras.layers.concatenate([ref_input, state_input])
            ctl_output = self.ctl_l(ctl_input)
        
            plant_input =  keras.layers.concatenate([state_input, ctl_output]) # X1_k, X2_k, U_k
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
        

def _validate_plant(plant, ctl):
     time = np.arange(0, 10., plant.dt)
     _sp = ut.step_vec(time, dt=8)
     _ctl = lambda X,t, k: [_sp[k]]
     X0 = [0., 0.]
     X, U = plant.sim(time, X0, _ctl)
     Xm = np.zeros((len(time), 2))
     Xm[0] = X0
     for k in range(1, len(time)):
         Xm[k] = ctl.plant_ann.predict(np.array([[Xm[k-1,0], Xm[k-1,1], _sp[k-1]]]))

     
     figure = robot_arm.plot(time, X, U); robot_arm.plot(time, Xm, figure=figure)
     plt.show()
     

def _validate_control(plant, ctl):
 
    def run_validation(time, sp, prefix, X0=[0.2, 0]):
        Xr, X, U = np.zeros((len(time),2)), np.zeros((len(time),2)), np.zeros((len(time),1))
        ctl.sp = sp
        X[0] = X0
        for k in range(0,len(time)-1):
            Xr[k+1], U[k] = ctl.get(k, X[k])
            X[k+1] = plant.disc_dyn(X[k], U[k])
        _unused, U[-1] = ctl.get(-1, X[-1])
        robot_arm.plot(time, X, U, Xr)
        plt.savefig('../docs/plots/mrc__robot_arm_{}.png'.format(prefix))
        plt.show()

    time =  np.arange(0., 8.05, plant.dt); sp = ut.step_vec(time, dt=8)
    run_validation( time, sp, 'test1')

    sp = ut.sine_vec(time)
    run_validation( time, sp, 'test2')

    sp = ut.sawtooth_vec(time)
    run_validation( time, sp, 'test3')
        
def main(train_plant=True, validate_plant=True, train_ctl=False):
    dt = 0.01
    plant = robot_arm.Plant(dt=dt)

    omega_ref, xi_ref, omega_err, xi_err = 6., 0.9, 20., 0.7
    ctl = Controller(dt, omega_ref, xi_ref, omega_err, xi_err)

    ctl.train_plant(plant, train=train_plant, make_training_set=False, epochs=10)
    if validate_plant: _validate_plant(plant, ctl)

    ctl.train_control(train=train_ctl, epochs=1000)
    _validate_control(plant, ctl)
     
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(train_plant=False, validate_plant=False, train_ctl=False)
