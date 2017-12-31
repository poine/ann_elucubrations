#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras, sklearn.neural_network
import utils as ut, lti_first_order as ltifo

import pdb

LOG = logging.getLogger('fo_lti_id_plant_feedforward_keras')

class ANN_Plant:
    ''' full state, delay 1 '''
    delay = 1
    x_km1, u_km1, input_size = range(3)
    def __init__(self):
        self.ann = keras.models.Sequential()
        self.ann.add(keras.layers.Dense(1, activation='linear', kernel_initializer='uniform', input_dim=2, use_bias=True))
        self.ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x_km1]   = X[i-1, 0]
            _input[i-self.delay, self.u_km1]   = U[i-1]
        return _input

    def fit(self, time, X, U):
        LOG.info(' Fitting Plant ANN on {} data points'.format(len(time)))
        LOG.info('  preparing training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:]
        LOG.info('  done. Now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        scaled_input = self.scaler.fit_transform(ann_input)
        LOG.info('  done. Now fitting set')
        self.ann.fit(scaled_input, ann_output, epochs=20, batch_size=32,  verbose=1)
        LOG.info('  done')
        #LOG.info('  score: {:f}'.format(self.ann.score(scaled_input , ann_output)))

    def save(self, filename):
        LOG.info('  saving ann to {}'.format(filename))
        with open(filename, "wb") as f:
            pickle.dump(self.scaler, f)
        self.ann.save(filename+'.h5')

    def load(self, filename):
        LOG.info(' Loading ann from {}'.format(filename))
        with open(filename, "rb") as f:
            self.scaler = pickle.load(f)
        self.ann = keras.models.load_model(filename+'.h5')

    def get(self, x_km1, u_km1):
        return self.ann.predict(self.scaler.transform([[x_km1[0], u_km1[0]]]))

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 1)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.get(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U



def main(make_training_set=True, train=True, test=True):
    training_traj_filename = '/tmp/frst_order_training_traj.pkl'
    ann_plant_filename = '/tmp/frst_order_plant_ann.pkl'

    tau, dt = 1., 1./100
    plant = ltifo.Plant(tau, dt)
    ctl = ut.CtlNone()
    ann = ANN_Plant()

    if train:
        if make_training_set:
            nsamples, max_nperiod = int(10*1e3), 10
            LOG.info('  Generating random setpoints')
            time, ctl.yc = ut.make_random_pulses(plant.dt, nsamples, max_nperiod=max_nperiod,  min_intensity=-10, max_intensity=10.)
            LOG.info('   done. Generated {} random setpoints'.format(len(time)))
            LOG.info('  Simulating trajectory ({} s)'.format(time[-1]))
            X0 = [0.]
            X, U = plant.sim(time, X0, ctl.get)
            LOG.info('   done')
            LOG.info('  Saving trajectory to {}'.format(training_traj_filename))
            desc = 'random setpoint trajectory. max_nperiod: {}'.format(max_nperiod)
            ut.save_trajectory(time, X, U, desc, training_traj_filename)
        else:
            LOG.info('  Loading trajectory from {}'.format(training_traj_filename))
            time, X, U, desc = ut.load_trajectory(training_traj_filename)
            LOG.info('     {} samples ({} s)'.format(len(time), time[-1]))
            LOG.info('     desc: {}'.format(desc))
     
        ann.fit(time, X, U)
        ann.save(ann_plant_filename)
    else:
        ann.load(ann_plant_filename)

    if test:
        print ann.ann.layers[0].get_weights() 
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
    main()
