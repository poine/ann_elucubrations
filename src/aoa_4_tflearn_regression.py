#! /usr/bin/env python
# -*- coding: utf-8 -*-


import math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

import tensorflow as tf, tflearn

import utils as ut, aoa_utils


def main(train=False, epochs=200, max_theta=ut.rad_of_deg(10)):
    Vp, Ap, theta, airspeed = aoa_utils.read_dataset('../data/aoa_cleaned_2.csv')
    
    inputs = tflearn.input_data(shape=[None, 2], dtype=tf.float32)
    true_output = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    
    net = tflearn.fully_connected(inputs, 8)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    net = tflearn.fully_connected(net, 8)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    out = tflearn.fully_connected(net, 2, activation='linear')
    
    if 0:
        my_optimizer =  tflearn.optimizers.SGD(learning_rate=0.001)
        def my_loss(y_pred, y_true):
            with tf.name_scope(None):
                return tf.nn.l2_loss(y_pred - y_true)
        #my_loss = tflearn.objectives.mean_square(out, true_output)
        net = tflearn.regression(out, true_output, optimizer=my_optimizer, loss=my_loss)
    else:
        net = tflearn.regression(out, optimizer='sgd', loss='mean_square', learning_rate=0.001, metric=None)

    model = tflearn.DNN(net, tensorboard_dir='/tmp/aoa_tflearn_logs/', best_checkpoint_path='/tmp/aoa_best', checkpoint_path='/tmp/aoa_current')
    if train:
        # we train with a cooked subset of the dataset
        Vp2, Ap2, theta2, airspeed2 = aoa_utils.cook_data(Vp, Ap, theta, airspeed, max_theta, -max_theta)
        training_input, training_output = np.vstack((Vp2, Ap2)).T, np.vstack((theta2, airspeed2)).T
        model.fit(training_input, training_output, n_epoch=epochs, batch_size=64, show_metric=True, validation_set=0.1)
        model.save('/tmp/aoa_final')
    else:
        model.load('/tmp/aoa_current-24624')
        #model.load('/tmp/aoa_final')

    
    # we test on the not stalled full dataset
    Vp3, Ap3, theta3, airspeed3 = aoa_utils.remove_stalled_data(Vp, Ap, theta, airspeed, max_theta, -max_theta)
    test_input, test_output =  np.vstack((Vp3, Ap3)).T, np.vstack((theta3, airspeed3)).T
    pred_out = model.predict(test_input)
    aoa_utils.plot_sequential(Vp3, Ap3, theta3, airspeed3, pred_out[:,0], pred_out[:,1])
    aoa_utils.plot_pred_err(pred_out,  test_output)
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    main(train=True)
