#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, pickle, scipy.signal
import sklearn.neural_network
import utils

import pdb
LOG = logging.getLogger('intro_learn_sine')


def model(_i): return np.sin(_i)

def make_ann(_input, _thruth, act, hidden_layers_shape):
    params = {
        'hidden_layer_sizes':hidden_layers_shape, # 
        'activation': act,                        # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        'solver': 'adam',                         # ‘lbfgs’, ‘sgd’, ‘adam’
        'verbose':False, 
        'random_state':1, 
        'max_iter':5000, 
        'tol':1e-30,
    }
    ann = sklearn.neural_network.MLPRegressor(**params)
    # train
    ann.fit(_input.reshape(-1, 1) , _thruth)
    print ann.coefs_, ann.intercepts_
    # test
    _output = ann.predict(_input.reshape(-1, 1))
    
    return _output

def plot(_i, _o):
    plt.plot(_i, _o)

def main():

    _input = np.arange(-math.pi, math.pi, 0.01)
    _thruth = model(_input)

    for act in ['identity', 'logistic', 'tanh', 'relu']:
        #left, bottom, right, top, wspace, hspace
        margins = (0.03, 0.04, 0.98, 0.93, 0.14, 0.15)
        utils.prepare_fig(figsize=(20.48, 10.24), margins=margins)
        plt.suptitle('Activation: {}'.format(act))
        nr, nc = 2,3
        for i, hl_shape in enumerate([(), (1,), (3,), (5,), (9,), (27,)]):
            _output = make_ann(_input, _thruth, act, hl_shape)
            ax = plt.subplot(nr, nc, i+1)
            plot(_input, _thruth)
            plot(_input, _output)
            utils.decorate(plt.gca(), title='hidden layer shape: {}'.format(hl_shape), legend=['thruth', 'predicted'], ylim=[-1, 1])
        plt.savefig('../docs/plots/intro__learn_sine__{}.png'.format(act))


    plt.show()
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
