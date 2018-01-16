#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt
import keras
import utils
import pdb

LOG = logging.getLogger('intro_activations')





def main():

    activations = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', 'tanh', 'linear']
    #activations = ['linear']
    w, b = np.array([[1]]), np.array([0])
    _input = np.arange(-10, 10, 0.1)
    for _act in activations:
        ann = keras.models.Sequential()
        ann.add(keras.layers.Dense(1, activation=_act, kernel_initializer='uniform', input_dim=1))
        ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        ann.layers[0].set_weights([w, b])
        _output = ann.predict(_input)
        plt.plot(_input, _output)

    utils.decorate(plt.gca(), 'Activation Functions (w=1, b=0)', 'input', 'output', activations)
    plt.savefig('../docs/plots/intro_activations.png', dpi=80)
    plt.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
