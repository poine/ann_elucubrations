#! /usr/bin/env python
# -*- coding: utf-8 -*-


import math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

import keras

import utils as ut, aoa_utils

def main(epochs=60, validation_split=0.2, display_training_history=True):
    Vp, Ap, theta, airspeed = aoa_utils.read_dataset('../data/aoa_cleaned_2.csv')

    _input_tensor = keras.layers.Input(shape=(2,))
    hls = ((32, 'l1'), (32, 'l2'))
    
    _t = _input_tensor
    for _s, _n in hls:
        _l = keras.layers.Dense(_s, activation='selu', kernel_initializer='uniform', name=_n)
        _t = _l(_t)
        
        
    _output_layer = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform',
                                       use_bias=True, name="output_layer")
    _ouput_tensor = _output_layer(_t)
    

    ann = keras.models.Model(inputs=_input_tensor, outputs=_ouput_tensor)
    ann.compile(loss=keras.losses.mean_squared_error, optimizer='adam')
    ann.summary()
    # we train with a cooked subset of the dataset
    Vp2, Ap2, theta2, airspeed2 = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    training_input, training_output = np.vstack((Vp2, Ap2)).T, np.vstack((theta2, airspeed2)).T

    history = ann.fit(training_input, training_output, epochs=epochs, batch_size=64,  verbose=1, shuffle=True,
                      validation_split=validation_split)
    if display_training_history:
        margins = (0.04, 0.07, 0.98, 0.93, 0.27, 0.2)
        figure = ut.prepare_fig(figsize=(20.48, 7.68), margins=margins)
        ax = plt.subplot(1,2,1); ut.decorate(ax, title='training loss'); plt.plot(history.history['loss'])
        ax = plt.subplot(1,2,2); ut.decorate(ax, title='validation loss'); plt.plot(history.history['val_loss'])
        ut.save_if('/tmp/aoa_training_history.png')

    
    # we test on the not stalled full dataset
    Vp3, Ap3, theta3, airspeed3 = aoa_utils.remove_stalled_data(Vp, Ap, theta, airspeed)
    test_input, test_output =  np.vstack((Vp3, Ap3)).T, np.vstack((theta3, airspeed3)).T
    pred_out = ann.predict(test_input)
    aoa_utils.plot_sequential(Vp3, Ap3, theta3, airspeed3, pred_out[:,0], pred_out[:,1])
    aoa_utils.plot_pred_err(pred_out,  test_output)
    plt.show()
    
        
if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    main()
