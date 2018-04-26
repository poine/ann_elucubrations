#! /usr/bin/env python
# -*- coding: utf-8 -*-


import math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import sklearn.neural_network

import utils as ut, aoa_utils
import pdb

def main():
    Vp, Ap, theta, airspeed = aoa_utils.read_dataset('../data/aoa_cleaned_2.csv')
     
    params = {
        'hidden_layer_sizes':(16, 16), # 
        'activation':'relu',         # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        'solver': 'lbfgs',           # ‘lbfgs’, ‘sgd’, ‘adam’
        'verbose': True, 
        'random_state':1, 
        'max_iter':500, 
        'tol':1e-12,
        'warm_start': True
    }
    ann = sklearn.neural_network.MLPRegressor(**params)
    # we train with a cooked subset of the dataset
    Vp2, Ap2, theta2, airspeed2 = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    tr_input, tr_output = np.vstack((Vp2, Ap2)).T, np.vstack((theta2, airspeed2)).T
    ann.fit(tr_input , tr_output)
    
    # we test on the full dataset - this is going to be bad for regions that are extrapolated
    te_input, te_output = np.vstack((Vp, Ap)).T, np.vstack((theta, airspeed)).T
    _pred_out = ann.predict(te_input)
    aoa_utils.plot_sequential(Vp, Ap, theta, airspeed, _pred_out[:,0], _pred_out[:,1])
    aoa_utils.plot_pred_err(_pred_out,  te_output)

    # we test on the not stalled full dataset
    Vp3, Ap3, theta3, airspeed3 = aoa_utils.remove_stalled_data(Vp, Ap, theta, airspeed)
    te_input, te_output = np.vstack((Vp3, Ap3)).T, np.vstack((theta3, airspeed3)).T
    _pred_out = ann.predict(te_input)
    aoa_utils.plot_sequential(Vp3, Ap3, theta3, airspeed3, _pred_out[:,0], _pred_out[:,1])
    aoa_utils.plot_pred_err(_pred_out,  te_output)
    
    plt.show()
    

if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    main()
