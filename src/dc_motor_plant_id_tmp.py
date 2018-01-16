#! /usr/bin/env python
# -*- coding: utf-8 -*-


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, pickle, scipy.signal
import sklearn.neural_network
import utils

import pdb

''' I am playing with some real data recorded on the motor '''



class ANN_PLANT:
    pass


class ANN_PLANT1(ANN_PLANT):
    ''' predict angle only '''
    delay = 4
    x_km1, x_km2, x_km3, x_km4, u_km1, u_km2, u_km3, u_km4, input_size = range(9)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(8,),   # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':True, 
            #'random_state':1, 
            'max_iter':5000, 
            'tol':1e-10,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x_km1] = X[i-1, 0]
            _input[i-self.delay, self.x_km2] = X[i-2, 0]
            _input[i-self.delay, self.x_km3] = X[i-3, 0]
            _input[i-self.delay, self.x_km4] = X[i-4, 0]
            _input[i-self.delay, self.u_km1] = U[i-1]
            _input[i-self.delay, self.u_km2] = U[i-2]
            _input[i-self.delay, self.u_km3] = U[i-3]
            _input[i-self.delay, self.u_km4] = U[i-4]
        return _input

    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:, 0]
        print(' done, now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(ann_input)
        scaled_input = self.scaler.transform(ann_input)
        print('fiting set')
        self.ann.fit(scaled_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(scaled_input , ann_output)))

    def get(self, _input):
        return self.ann.predict(self.scaler.transform([_input]))
    
    def sim_with_input_vec(self, time, U):
        X =  np.zeros((len(time), 1))
        for i in range(self.delay, len(time)):
            X[i] = self.get((X[i-1, 0], X[i-2, 0], X[i-3, 0], X[i-4, 0], U[i-1], U[i-2], U[i-3], U[i-4]))
        return X


class ANN_PLANT2(ANN_PLANT):
    ''' 
      Predict angle and rotational velocity
      trained with angle
    '''
    delay = 4
    x_km1, x_km2, x_km3, x_km4, u_km1, u_km2, u_km3, u_km4, input_size = range(9)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(4,),   # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':True, 
            'random_state':1, 
            'max_iter':5000, 
            'tol':1e-20,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x_km1] = X[i-1, 0]
            _input[i-self.delay, self.x_km2] = X[i-2, 0]
            _input[i-self.delay, self.x_km3] = X[i-3, 0]
            _input[i-self.delay, self.x_km4] = X[i-4, 0]
            _input[i-self.delay, self.u_km1] = U[i-1]
            _input[i-self.delay, self.u_km2] = U[i-2]
            _input[i-self.delay, self.u_km3] = U[i-3]
            _input[i-self.delay, self.u_km4] = U[i-4]
        return _input
    
    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:, :2]
        print(' done, now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(ann_input)
        scaled_input = self.scaler.transform(ann_input)
        print('fiting set')
        self.ann.fit(scaled_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(scaled_input , ann_output)))

    def get(self, _input):
        return self.ann.predict(self.scaler.transform([_input]))
    
    def sim_with_input_vec(self, time, U):
        X =  np.zeros((len(time), 3))
        for i in range(self.delay, len(time)):
            X[i,:2] = self.get((X[i-1, 0], X[i-2, 0], X[i-3, 0], X[i-4, 0], U[i-1], U[i-2], U[i-3], U[i-4]))
        return X

class ANN_PLANT2bis(ANN_PLANT):
    ''' 
      Predict angle and rotational velocity
      trained with angle and rotational velocity
    '''
    delay = 3
    x_km1, x_km2, x_km3, xd_km1, xd_km2, u_km1, input_size = range(7)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(4,),   # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':True, 
            'random_state':1, 
            'max_iter':5000, 
            'tol':1e-20,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x_km1] = X[i-1, 0]
            _input[i-self.delay, self.x_km2] = X[i-2, 0]
            _input[i-self.delay, self.x_km3] = X[i-3, 0]
            _input[i-self.delay, self.xd_km1] = X[i-1, 1]
            _input[i-self.delay, self.xd_km2] = X[i-2, 1]
            _input[i-self.delay, self.u_km1] = U[i-1]
        return _input
    
    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:, :2]
        print(' done, now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(ann_input)
        scaled_input = self.scaler.transform(ann_input)
        print('fiting set')
        self.ann.fit(scaled_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(scaled_input , ann_output)))

    def get(self, _input):
        return self.ann.predict(self.scaler.transform([_input]))
    
    def sim_with_input_vec(self, time, U):
        X =  np.zeros((len(time), 3))
        for i in range(self.delay, len(time)):
            X[i,:2] = self.get((X[i-1, 0], X[i-2, 0], X[i-3, 0], X[i-1, 1], X[i-2, 1], U[i-1]))
        return X



class ANN_PLANT3(ANN_PLANT):
    ''' 
      Predict angle and rotational velocity
      trained with angle, rotational velocity, rotational accel
    '''
    delay = 4
    x_km1, x_km2, x_km3, x_km4, xd_km1, xd_km2, xd_km3, xdd_km1, xdd_km2, u_km1, input_size = range(11)

    def __init__(self):
        params = {
            'hidden_layer_sizes':(8,),  # 
            'activation':'identity',     # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            'solver': 'adam',            # ‘lbfgs’, ‘sgd’, ‘adam’
            'verbose':True, 
            'random_state':1, 
            'max_iter':10000, 
            'tol':1e-12,
            'warm_start': True
        }
        self.ann = sklearn.neural_network.MLPRegressor(**params)

    def make_input(self, X, U):
        _input = np.zeros(( len(X)- self.delay, self.input_size))
        for i in range(self.delay, len(X)):
            _input[i-self.delay, self.x_km1] = X[i-1, 0]
            _input[i-self.delay, self.x_km2] = X[i-2, 0]
            _input[i-self.delay, self.x_km3] = X[i-3, 0]
            _input[i-self.delay, self.x_km4] = X[i-4, 0]
            _input[i-self.delay, self.xd_km1] = X[i-1, 1]
            _input[i-self.delay, self.xd_km2] = X[i-2, 1]
            _input[i-self.delay, self.xd_km3] = X[i-3, 1]
            _input[i-self.delay, self.xdd_km1] = X[i-1, 2]
            _input[i-self.delay, self.xdd_km2] = X[i-2, 2]
            _input[i-self.delay, self.u_km1] = U[i-1]
        return _input
    
    def fit(self, time, X, U):  
        print('building training set')
        ann_input, ann_output = self.make_input(X, U), X[self.delay:, :3]
        print(' done, now scaling training set')
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(ann_input)
        scaled_input = self.scaler.transform(ann_input)
        print('fiting set')
        self.ann.fit(scaled_input , ann_output)
        print(' done')
        print('score: {:e}'.format(self.ann.score(scaled_input , ann_output)))

    def get(self, _input):
        return self.ann.predict(self.scaler.transform([_input]))
    
    def sim_with_input_vec(self, time, U):
        X =  np.zeros((len(time), 3))
        for i in range(self.delay, len(time)):
            X[i,:] = self.get((X[i-1, 0], X[i-2, 0], X[i-3, 0], X[i-4, 0], X[i-1, 1], X[i-2, 1], X[i-3, 1], X[i-1, 2], X[i-2, 2], U[i-1]))
        return X

    


    
    
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


def plot_err(time, X, Xe, figure=None, window_title="error"):
    margins=(0.04, 0.05, 0.98, 0.96, 0.20, 0.34)
    figure = utils.prepare_fig(figure, window_title, figsize=(0.75*20.48, 0.75*10.24), margins=margins)
    ax = plt.subplot(2, 1, 1)
    err_x = X[:,0]-Xe[:,0]
    plt.plot(time,  err_x)
    mu, sigma = np.mean(err_x), np.std(err_x)
    plt.text(0.5, 0.5, '$\mu$ {:.1f} $\sigma$ {:.1f}'.format(mu, sigma), transform=ax.transAxes, fontsize=16)
    utils.decorate(ax, title='err $\\theta$')
    
        
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


def main(train_filename, test_filename, ann_kind):

    time, X, U = load_traj(train_filename, mode='filter')
    plot(time, X, U, window_title='training set')

    ann =  ann_kind()
    ann.fit(time, X, U)

    time_test, X_test, U_test = load_traj(test_filename)
    Xe_test = ann.sim_with_input_vec(time_test, U_test)

    figure = plot(time_test, X_test, U_test)
    plot(time_test, Xe_test, None, figure,'test set')

    plot_err(time_test, X_test, Xe_test)
    
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ann_kind = ANN_PLANT2bis
    main('../data/motor_log_sine_sweep.pkl', '../data/motor_log_sine.pkl', ann_kind)
    #main('motor_log_random_pulses.pkl', 'motor_log_sine.pkl', ann_kind)
    #main('motor_log_sine_sweep_2.pkl', 'motor_log_sine.pkl', ann_kind)
