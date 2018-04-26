#! /usr/bin/env python
# -*- coding: utf-8 -*-


''' 
   this is a (firtst order LTI) plant experiment with RNM
   I am not able to train it yet, but when i force the weights, i get the output i am hopping for, that is
   X_kp1 = a x_k + b u_k

   https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
   https://pjb.primecdn.net/pics/original/06/06e8eb62.jpg
'''


import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras, sklearn.neural_network
import utils as ut, lti_first_order as fo_lti

import pdb

LOG = logging.getLogger('fo_lti_plant_if_rnn_keras')

class ANN_Plant:
    ''' full state, delay 1 '''
    delay = 1
    x_km1, u_km1, input_size = range(3)
    def __init__(self):
        self.model =  keras.models.Sequential()
        input_dim = 1
        timesteps = 2
        batch_size = 1
        self.model.add(keras.layers.SimpleRNN(units=1,
                                              input_shape=(timesteps, input_dim),
                                              batch_size=batch_size,
                                              stateful=True,
                                              use_bias=False,
                                              return_state=False,
                                              unroll=False,
                                              activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()


    def fit(self, time, X, U):
        _bs= 1
        _len = 2#_bs*336
        _input, _output = U[:_len].reshape((1, 2, 1)), X[2:_len+1]
        for i in range(15):
            self.model.layers[0].reset_states(states=np.array([X[0]]))
            self.model.fit(_input, _output, epochs=1,  verbose=1, batch_size=_bs, shuffle=False)
            self.model.reset_states()

        print self.model.layers[0].get_weights()
        

    def force(self, a, b):
        self.model.layers[0].set_weights([np.array([[b]]), np.array([[a]])])
        print self.model.layers[0].get_weights()
    

    def save(self, filename):
        LOG.info('  saving ann to {}'.format(filename))   

    def summary(self):
        pass

    def sim(self, time, X0, ctl):
        self.model.layers[0].reset_states(states=np.array([X0]))
        if 0: # no loop
            U = np.array([ctl(None, i) for i in range(len(time))])
            X = self.model.predict(U.reshape((len(time), 1, 1)), batch_size=1)
        else: # with loop
            X, U = np.zeros((len(time), 1)),  np.zeros((len(time), 1))
            X[0] = X0
            for i in range(1, len(time)):
                U[i-1] = ctl(X[i-1], i-1)
                X[i] = self.model.predict(U[i-1].reshape((1,1,1)))
            U[-1] = U[-2]
        #pdb.set_trace()
        return X, U

def main(make_training_set=True, train=True, test=True):
    ann_plant_filename = '/tmp/fo_lti_plant_id_rnn.pkl'

    tau, dt= 0.8, 0.01
    plant, ctl = fo_lti.Plant(tau, dt), ut.CtlNone()
    ann = ANN_Plant()
    if train:
        time, X, U, desc = fo_lti.make_or_load_training_set(plant, ctl, make_training_set)
        ann.fit(time, X, U)
        #ann.force(plant.ad, plant.bd)
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
        fo_lti.plot(time, X1, U1)
        fo_lti.plot(time, X2, U2)
        plt.subplot(2,1,1); plt.legend(['plant','ANN'])
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main(make_training_set=True, train=True, test=True)
