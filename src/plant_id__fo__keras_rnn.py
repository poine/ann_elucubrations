#! /usr/bin/env python
# -*- coding: utf-8 -*-


''' 
   this is a (firtst order LTI) plant identification experiment with Keras RNM
'''

import logging, timeit, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt, pickle
import keras
import utils as ut, lti_first_order as fo


import  pdb


def ident_stateless_RNN(plant, make_training_set=True, train=True):
    # train a RNN stateless model
    filename = "/tmp/fo_plant_rnn.h5"
    if train:
        time, X, U, desc = fo.make_or_load_training_set(plant, ut.CtlNone(), make_training_set)
        input_tensor = keras.layers.Input((1,1), name="ctl_input")
        initial_state_tensor = keras.layers.Input((1,1), name="state_input")
        rnn_layer = keras.layers.SimpleRNN(units=1,
                                           input_shape=(1, 1),
                                           batch_size=1,
                                           stateful=False,
                                           use_bias=False,
                                           return_state=False,
                                           unroll=False,
                                           activation='linear',
                                           name="rnn_layer")
        output_tensor = rnn_layer(input_tensor, initial_state=initial_state_tensor)
        model = keras.models.Model([input_tensor, initial_state_tensor], output_tensor)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        _input, _output = [U[:-1].reshape((len(U)-1, 1, 1)), X[:-1].reshape((len(X)-1, 1, 1))], X[1:]
        model.fit(_input, _output, epochs=3,  verbose=1, batch_size=1)
        model.save(filename)
    else:
        model = keras.models.load_model(filename)

    weights = model.get_layer(name="rnn_layer").get_weights()
    print('rnn weights: {}'.format(weights))
    return weights


def ident_stateful_RNN(plant, make_training_set=True, train=True):
    # train a RNN stateful model - I don't know how to do that :(
    time, X, U, desc = fo.make_or_load_training_set(plant, ut.CtlNone(), make_training_set)

    seq_size = 10
    rnn_layer = keras.layers.SimpleRNN(units=1,
                                       input_shape=(1, 1),
                                       batch_size=2,
                                       stateful=True,
                                       use_bias=False,
                                       return_state=False,
                                       unroll=False,
                                       activation='linear',
                                       name="rnn_layer")
    model = keras.models.Sequential()
    model.add(rnn_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    _input = np.zeros(())
    _output = np.zeros(())
    model.fit(_input, _output, epochs=3,  verbose=1
              
              
def test_stateful_RNN(plant, weights):
    # Test a stateful model
    rnn_layer = keras.layers.SimpleRNN(units=1,
                                       input_shape=(1, 1),
                                       batch_size=1,
                                       stateful=True,
                                       use_bias=False,
                                       return_state=False,
                                       unroll=False,
                                       activation='linear',
                                       name="rnn_layer")
    model = keras.models.Sequential()
    model.add(rnn_layer)
    model.get_layer(name="rnn_layer").set_weights(weights)
    model.summary()

    time, ctl =  np.arange(0., 15.05, plant.dt), ut.CtlNone()
    ctl.yc, X0 = ut.step_vec(time, dt=8), [0.5]
    Xp, U = plant.sim(time, X0, ctl.get)
    model.get_layer(name="rnn_layer").reset_states(states=np.array([X0]))
    Xm = model.predict(U.reshape((len(U), 1, 1)), batch_size=1)
    fo.plot(time, Xp, U, Xm)
    plt.show()
    
def main():
    plant = fo.Plant()
    #weights = ident_stateless_RNN(plant, make_training_set=False, train=False)
    weights = ident_stateful_RNN(plant, make_training_set=False, train=False)
    #test_stateful_RNN(plant, weights)

    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
