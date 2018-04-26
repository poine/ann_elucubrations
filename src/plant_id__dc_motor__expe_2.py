#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
   trying dc motor real data with sklearn
'''


import logging, math, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf, tflearn

import utils, plant_id__dc_motor__expe_1 as ex1
import pdb

''' 
Predict angle and rotational velocity
trained with angle and rotational velocity
'''
class ANN1:

    delay = 3
    x_km1, x_km2, x_km3, xd_km1, xd_km2, u_km1, u_km2, u_km3, input_size = range(9)
    xk, xkd, output_size = range(3)

    @staticmethod
    def prepare_training_set(time, X, U):
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
        return _input, _output

    @staticmethod
    def train(time, X, U, epochs=500):
        training_input, training_output = ANN1.prepare_training_set(time, X, U)
        
        input_tensor = tflearn.input_data(shape=[None, ANN1.input_size], dtype=tf.float32)
        true_output_tensor = tf.placeholder(shape=(None, ANN1.output_size), dtype=tf.float32)

        net = tflearn.fully_connected(input_tensor, 8)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        net = tflearn.fully_connected(net, 8)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, 2, activation='linear')

        net = tflearn.regression(out, optimizer='sgd', loss='mean_square', learning_rate=0.001, metric=None)

        model = tflearn.DNN(net, tensorboard_dir='/tmp/dc_motor_tflearn_logs/', best_checkpoint_path='/tmp/dc_motor_best', checkpoint_path='/tmp/dc_motor_current')

        model.fit(training_input, training_output, n_epoch=epochs, batch_size=64, show_metric=True, validation_set=0.1)

        
def main(train_filename, test_filename, epochs):
    timet, Xt, Ut = ex1.load_traj(train_filename, mode='filter')
    #ex1.plot(timet, Xt, Ut, window_title='training set')
    ANN1.train(timet, Xt, Ut)
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_file, test_file = '../data/motor_log_sine_sweep.pkl', '../data/motor_log_sine.pkl'
    main(train_file, test_file, epochs=3500)
