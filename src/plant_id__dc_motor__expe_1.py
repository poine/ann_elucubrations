#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, pickle, scipy.signal
import keras
import utils

import pdb

''' I am playing with some real data recorded on the motor '''
def load_traj(filename, mode='gradient'):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    time, duty, angle = data[:,0], data[:,1], data[:,2]
    X = np.zeros((len(time), 3))
    X[:,0] = angle
    if mode=='gradient':
        # it looks like gradient is flunked.... in 1.11
        # https://docs.scipy.org/doc/numpy-1.11.0/reference/generated/numpy.gradient.html
        # should be better in 1.13, soon...
        X[:,1] = np.gradient(angle,  0.01, edge_order=2)#time)
        X[:,2] = np.gradient(X[:,1], 0.01, edge_order=2)#time)
    else:
        rvel_filtered = scipy.signal.savgol_filter(angle, window_length=3, polyorder=1, deriv=1, delta=0.01, mode='nearest')
        X[:,1] = rvel_filtered
        racc_filtered = scipy.signal.savgol_filter(angle, window_length=7, polyorder=2, deriv=2, delta=0.01, mode='nearest')
        X[:,2] = racc_filtered
    return time, X, duty

def plot(time, X, U, figure=None, window_title="trajectory"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = utils.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    _nc, _nr = X.shape
    ax = plt.subplot(4, 1, 1)
    plt.plot(time, X[:,0])
    utils.decorate(ax, title='$\\theta$')
    if _nr > 1:
        ax = plt.subplot(4, 1, 2)
        plt.plot(time, X[:,1])
        utils.decorate(ax, title='$\omega$')
    if _nr > 2:   
        ax = plt.subplot(4, 1, 3)
        plt.plot(time, X[:,2])
        utils.decorate(ax, title='$\dot{\omega}$', ylim=[-15000, 15000])
    if U is not None:
        ax = plt.subplot(4, 1, 4)
        plt.plot(time, U)
        utils.decorate(ax, title='duty')
    return figure


''' 
Predict angle and rotational velocity
trained with angle and rotational velocity
'''
class ANN1:

    delay = 3
    x_km1, x_km2, x_km3, xd_km1, xd_km2, u_km1, u_km2, u_km3, input_size = range(9)
    xk, xkd, output_size = range(3)

    @staticmethod
    def train(time, X, U, epochs=500):
        plant_input = keras.layers.Input(shape=(ANN1.input_size,))
        plant_layer1 = keras.layers.Dense(4, activation='linear', kernel_initializer='uniform', use_bias=True)
        plant_layer2 = keras.layers.Dense(ANN1.output_size, activation='linear', kernel_initializer='uniform', use_bias=True)
        plant_output = plant_layer2(plant_layer1(plant_input))

        ANN1.model = keras.models.Model(inputs=plant_input, outputs=plant_output)
        ANN1.model.compile(loss='mean_squared_error', optimizer='adam') # adam, sgd, RMSprop, Adagrad, Adadelta, Adamax, Nadam
        ANN1.model.summary()
        
        _input =  np.zeros(( len(X)- ANN1.delay, ANN1.input_size))
        for i in range(ANN1.delay, len(X)):
            _input[i-ANN1.delay, ANN1.x_km1]  = X[i-1, 0]
            _input[i-ANN1.delay, ANN1.x_km2]  = X[i-2, 0]
            _input[i-ANN1.delay, ANN1.x_km3]  = X[i-3, 0]
            _input[i-ANN1.delay, ANN1.xd_km1] = X[i-1, 1]
            _input[i-ANN1.delay, ANN1.xd_km2] = X[i-2, 1]
            _input[i-ANN1.delay, ANN1.u_km1]  = U[i-1]
            _input[i-ANN1.delay, ANN1.u_km2]  = U[i-2]
            _input[i-ANN1.delay, ANN1.u_km3]  = U[i-3]
        _output = X[ANN1.delay:, :2]
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
        history = ANN1.model.fit(_input, _output, epochs=epochs, batch_size=32,
                                 verbose=1, shuffle=True)#, validation_split=0.1, callbacks=[early_stopping])
        filename = '/tmp/plant_id__dc_motor__expe_1.h5'
        ANN1.model.save(filename)

        plt.plot(history.history['loss']); plt.title("loss")
        plt.show()

    @staticmethod
    def test(time, X, U):
        print ANN1.model.get_layer(name='dense_1').get_weights()
        print ANN1.model.get_layer(name='dense_2').get_weights()
        Xm =  np.zeros((len(time), ANN1.output_size))
        if 0:
            Xm[:3,0] = X[:3,0]
            Xm[:2,1] = X[:2,1]
        for k in range(ANN1.delay, len(time)):
            _input = np.array([[Xm[k-1, 0], Xm[k-2, 0], Xm[k-3, 0], Xm[k-1, 1], Xm[k-2, 1], U[k-1], U[k-2], U[k-3]]])
            Xm[k,:] = ANN1.model.predict(_input)
        figure = plot(time, X, U)
        plot(time, Xm, U, figure=figure, window_title='testing_set')


        
def main(train_filename, test_filename, epochs):
    if type(train_filename) is list:
        trajs = [load_traj(_t, mode="filter") for _t in train_filename]
        timet = np.concatenate([ _time for (_time, _X, _U) in trajs])
        Xt = np.concatenate([ _X for (_time, _X, _U) in trajs])
        Ut = np.concatenate([ _U for (_time, _X, _U) in trajs])
    else:
        timet, Xt, Ut = load_traj(train_filename, mode='filter')

    #plot(timet, Xt, Ut, window_title='training set')
    ANN1.train(timet, Xt, Ut, epochs)
    timee, Xe, Ue = load_traj(test_filename, mode='filter')
    ANN1.test(timee, Xe, Ue)
    plt.show()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main('../data/motor_log_sine_sweep.pkl', '../data/motor_log_sine.pkl', epochs=3500)
    #main('../data/motor_log_sine_sweep_2.pkl', '../data/motor_log_sine.pkl', epochs=100)
    #main('../data/motor_log_random_pulses.pkl', '../data/motor_log_sine.pkl', epochs=100)

    #all_trainigs = ['../data/motor_log_random_pulses.pkl', '../data/motor_log_sine_sweep.pkl' , '../data/motor_log_sine_sweep_2.pkl']
    #main(all_trainigs, '../data/motor_log_sine.pkl', epochs=100)
