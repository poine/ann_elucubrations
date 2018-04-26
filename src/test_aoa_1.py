#! /usr/bin/env python
# -*- coding: utf-8 -*-

#
# A_p is the pressure sensor measuring the AoA
# V_p is the pressure sensor measuring the airspeed
# Theta is the pitch angle coming from IMU, which is equal to real AoA in wind tunnel
# Airspeed is the value we set the wind tunnel speed during the experiments.
#

import logging, timeit, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import keras
import tensorflow as tf, tflearn

import utils as ut, aoa_utils
import pdb


 

def plot_pred_err(pred_out,  test_output):
    plt.figure()
    pred_err = (pred_out - test_output)
    mus, sigmas = np.mean(pred_err, axis=0), np.std(pred_err, axis=0)
    ax = plt.subplot(1,2,1)
    plt.hist(ut.deg_of_rad(pred_err[:,0]), bins=100)
    ut.decorate(ax, title='Theta err', xlab='deg', legend=['$\mu$ {:.3f} deg $\sigma$ {:.3f} deg'.format(ut.deg_of_rad(mus[0]), ut.deg_of_rad(sigmas[0]))])
    ax = plt.subplot(1,2,2)
    plt.hist(pred_err[:,1], bins=100)
    ut.decorate(ax, title='Airspeed err', xlab='m/s', legend=['$\mu$ {:.3f} m/s $\sigma$ {:.3f} m/s'.format(mus[1], sigmas[1])])

    
def test_sklearn(Vp, Ap, theta, airspeed):
    params = {
        'hidden_layer_sizes':(100, 100, 10),     # 
        'activation':'relu',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        'solver': 'lbfgs',            # ‘lbfgs’, ‘sgd’, ‘adam’
        'verbose': True, 
        'random_state':1, 
        'max_iter':500, 
        'tol':1e-12,
        'warm_start': True
    }
    ann = sklearn.neural_network.MLPRegressor(**params)
    _input = np.vstack((Vp, Ap)).T
    _output = np.vstack((theta, airspeed)).T
    ann.fit(_input , _output)
    _pred_out = ann.predict(_input)
    plot(Vp, Ap, theta, airspeed, _pred_out[:,0], _pred_out[:,1])




    

from keras import backend as K
def test_keras(Vp, Ap, theta, airspeed, epochs=800, validation_split=0.2, display_training_history=True):
    _input_tensor = keras.layers.Input(shape=(2,))
    act_reg = 0.0001
    hls = ((32, 'l1'),
           (32, 'l2'),
    )
    
#                             kernel_regularizer=keras.regularizers.l2(0.001),
#                             bias_regularizer=keras.regularizers.l2(0.01),                             
#                             activity_regularizer=keras.regularizers.l2(act_reg),

    _t = _input_tensor
    for _s, _n in hls:
        _l = keras.layers.Dense(_s, activation='selu', kernel_initializer='uniform', name=_n)
        _t = _l(_t)
    
    
    _output_layer = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform',
                                       use_bias=True, name="output_layer")
    _ouput_tensor = _output_layer(_t)

    ann = keras.models.Model(inputs=_input_tensor, outputs=_ouput_tensor)
    def my_weighted_loss(y_true, y_pred):
        err = y_pred - y_true
        W = K.variable(np.array([[0.1, 0], [0, 0.9]]))
        return K.mean( K.square(err),axis=-1  )

    if 0:
        y_a = K.variable(np.random.random((6, 7)))
        y_b = K.variable(np.random.random((6, 7)))
        output = weighted_loss(y_a, y_b)
        result = K.eval(output)
        print result
        pdb.set_trace()
    #ann.compile(loss=keras.losses.mean_squared_error, optimizer='adam')
    #ann.compile(loss=keras.losses.mean_absolute_error, optimizer='adam')
    #ann.compile(loss=keras.losses.mean_squared_logarithmic_error, optimizer='adam')
    ann.compile(loss=my_weighted_loss, optimizer='adam')
    ann.summary()

    
    Vp, Ap, theta, airspeed = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    
    training_input = np.vstack((Vp, Ap)).T
    training_output = np.vstack((theta, airspeed)).T
    history = ann.fit(training_input, training_output, epochs=epochs, batch_size=64,  verbose=1, shuffle=True,
                      validation_split=validation_split)
    if display_training_history:
        margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
        figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
        ax = plt.subplot(1,2,1); ut.decorate(ax, title='training loss'); plt.plot(history.history['loss'])
        ax = plt.subplot(1,2,2); ut.decorate(ax, title='validation loss'); plt.plot(history.history['val_loss'])
        ut.save_if('/tmp/aoa_training_history.png')


    test_input = training_input
    test_output = training_output
    pred_out = ann.predict(test_input)
    #pdb.set_trace()
    plt.figure()
    plot(Vp, Ap, theta, airspeed, pred_out[:,0], pred_out[:,1])
    if 1:
        plot_pred_err(pred_out,  test_output)



def test_tflearn(Vp, Ap, theta, airspeed, epochs=800):
    inputs = tflearn.input_data(shape=[None, 2], dtype=tf.float32)
    true_output = tf.placeholder(shape=(None, 2), dtype=tf.float32)

    net = tflearn.fully_connected(inputs, 8)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    net = tflearn.fully_connected(inputs, 8)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    #w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, 2, activation='linear')#, weights_init=w_init)

    if 1:
        my_optimizer =  tflearn.optimizers.SGD(learning_rate=0.001)
        def my_loss(y_pred, y_true):
            with tf.name_scope(None):
                return tf.nn.l2_loss(y_pred - y_true)
        my_loss = tflearn.objectives.mean_square(out, true_output)
        net = tflearn.regression(out, true_output, optimizer=my_optimizer, loss=my_loss)
    else:
        net = tflearn.regression(out, optimizer='sgd', loss='mean_square', learning_rate=0.001)
    model = tflearn.DNN(net, tensorboard_dir='/tmp/aoa_tflearn_logs/')#, checkpoint_path='/tmp/aoa_tflearn_chekpoints/')

    Vp, Ap, theta, airspeed = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    
    training_input = np.vstack((Vp, Ap)).T
    training_output = np.vstack((theta, airspeed)).T
    model.fit(training_input, training_output, n_epoch=epochs, batch_size=64, show_metric=False)

    


    
    test_input = training_input
    test_output = training_output
    pred_out = model.predict(test_input)

    plt.figure()
    plot(Vp, Ap, theta, airspeed, pred_out[:,0], pred_out[:,1])
    if 1:
        plot_pred_err(pred_out,  test_output)

    

    # Initializing the variables
    #init = tf.global_variables_initializer()
    #with tf.Session() as sess:
    #    sess.run(init)
        
        
        
def main():
    Vp, Ap, theta, airspeed = aoa_utils.read_dataset('../data/aoa_cleaned_2.csv')
    
    #analyze_measurements(Vp, Ap)

    #Vp2, Ap2, theta2, airspeed2 = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    #analyze_measurements(Vp2, Ap2)
    
    #test_sklearn(Vp, Ap, theta, airspeed)
    #test_keras(Vp, Ap, theta, airspeed, epochs=40, validation_split=0.1)
    test_tflearn(Vp, Ap, theta, airspeed, epochs=200)
    plt.show()
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
