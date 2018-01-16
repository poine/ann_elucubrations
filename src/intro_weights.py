#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt
import keras
import utils
import pdb

LOG = logging.getLogger('intro_weight_effect')


def run_single_neuron(ann, w, b, _input):
    ann.layers[0].set_weights([np.array([w]), np.array(b)])
    return ann.predict(_input)


def plot_single_neuron():
    _input = np.arange(-2, 2, 0.1)
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(1, activation='tanh', input_dim=1))
    weights = [[0.5, 0],
               [1.,  0],
               [1.5, 0]]
    ax = plt.subplot(1,2,1)
    for w,b in weights:
        _output = run_single_neuron(ann, w, b, _input)
        plt.plot(_input, _output)
    utils.decorate(ax, 'Weights (b=0)', 'input', 'output', ['w: {}'.format(w) for w,_ in weights])

    weights = [[1., 0.],
               [1., 1.0],
               [1., 2.0]]
    ax = plt.subplot(1,2,2)
    for w,b in weights:
        _output = run_single_neuron(ann, w, b, _input)
        plt.plot(_input, _output)
    utils.decorate(ax, 'Biases (w=1)', 'input', 'output', ['b: {}'.format(b) for w,b in weights]) 

    plt.savefig('../docs/plots/intro_weights.png', dpi=80)


def plot_dual_neuron_network():
    _input = np.arange(-5, 5, 0.1)
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(2, activation='tanh', input_dim=1))
    ann.add(keras.layers.Dense(1, activation='linear'))
    #pdb.set_trace()
    ann.layers[1].set_weights([np.array([[-1], [1]]), np.array([0])])
    weights = [[[1.,  1.], [-1., 1.]],
               [[1.,  1.], [-0.5, 0.5]],
               [[1.,  1.], [-2.,  4.]],
               [[1.,  2.], [-1., 3]]]
    for w,b in weights:
        _output = run_single_neuron(ann, w, b, _input)
        plt.plot(_input, _output)
    utils.decorate(plt.gca(), '3 Neurons, 2 layers Network', 'input', 'output')#, ['b: {}'.format(b) for w,b in weights]) 
    plt.savefig('../docs/plots/intro_network_weights.png', dpi=80)

def main():
    #plot_single_neuron()
    plot_dual_neuron_network()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
