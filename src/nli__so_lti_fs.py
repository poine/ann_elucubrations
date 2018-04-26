#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
An attempt at full state NLI control. I am not sure what I am doing here...

TODO
'''


import logging, timeit, math, numpy as np, matplotlib.pyplot as plt, scipy
import keras, control
import utils as ut, so_lti

import pdb
LOG = logging.getLogger('nli__so_lti')

class Reference:
    def __init__(self, omega, xi, dt):
        txio, o2 = 2*xi*omega, omega**2
        c_tf = control.tf([o2], [1, txio, o2])
        d_tf = c_tf.sample(dt, method='zoh')
        self.a1, self.a0 = d_tf.num[0][0]
        self.b2, self.b1, self.b0 = d_tf.den[0][0]
        print self.b1, self.b0
        self.y_k, self.y_km1, self.yc_km1 = 0., 0., 0.

    def get(self, yc_k):
        y_kp1 = -self.b1*self.y_k -self.b0*self.y_km1 +self.a1*yc_k +self.a0*self.yc_km1
        self.y_km1 = self.y_k
        self.y_k = y_kp1
        self.yc_km1 = yc_k
        return self.y_k


    
class NLIController:
    def __init__(self, omega_ref, xi_ref, omega_err, xi_err, dt):
        self.ref = Reference(omega_ref, xi_ref, dt)
        ann_filename = '/tmp/so_lti__plant_id__ctaf__sklearn.pkl'
        ann = keras.models.load_model(ann_filename+'.h5')
        _fl, _gl = [ann.get_layer(name=_n) for _n in ['f', 'g']]
        LOG.info('f: {}'.format(_fl.get_weights()))
        LOG.info('g: {}'.format(_gl.get_weights()))
        _i = keras.layers.Input(shape=(3,)) # x_km1, x_km2, u_km2
        self.f = keras.models.Model(inputs=_i, outputs=_fl(_i))
        self.g = keras.models.Model(inputs=_i, outputs=_gl(_i))

        # roots of continuous time error caracteristic polynomial
        c_r = np.roots([1, 2*xi_err*omega_err, omega_err**2])
        if 1: # check
            Ac = np.array([[0, 1],[-omega_err**2, -2*xi_err*omega_err]])
            Ad = scipy.linalg.expm(dt*Ac)
            ceva, ceve = np.linalg.eig(Ac)
            deva, deve = np.linalg.eig(Ad)
        # roots of discrete time error caracteristic polynomial
        d_r = np.exp(c_r*dt)
        # coefficents of discrete time error caracteristic polynomial
        d_p = np.poly(d_r)
        #self.be1, self.be0 = 0.001, 0.002
        self.be1, self.be0 = d_p[1:]
        print self.be1, self.be0
        if 0: # check
            _t = np.arange(0, 5, dt)
            _y = np.zeros(len(_t))
            _y[0] = _y[1] = 1
            for k in range(1,len(_t)-1):
                _y[k+1] = -self.be1*_y[k]-self.be0*_y[k-1]
            plt.plot(_t, _y)
            plt.show()
            
        #pdb.set_trace()
        
    def get(self, yc_k, y_k, y_km1, u_km1):
        _i = np.array([[y_k, y_km1, u_km1]])
        #_f, _g = [_m.predict(_i) for _m in [self.f, self.g]]
        _f, _g = self.thruth(y_k, y_km1, u_km1)
        
        _yr_k, _yr_km1, _yr_kp1 = self.ref.y_k, self.ref.y_km1, self.ref.get(yc_k)
        _e_k, _e_km1 = y_k - _yr_k, y_km1 - _yr_km1
        _u_k = (_yr_kp1 -_f -self.be1*_e_k -self.be0*_e_km1)/_g

        return _yr_kp1, _u_k


    def force_truth(self, plant):
        self.a0, self.a1, self.b0, self.b1 = plant.a0, plant.a1, plant.b0, plant.b1

    def thruth(self, y_k, y_km1, u_km1):
        f = -self.b1*y_k -self.b0*y_km1 +self.a0*u_km1
        g = self.a1
        return f, g
    
    
def main():
    omega_plant, xi_plant, dt = 3., 0.2, 0.01
    plant = so_lti.CCPlant(omega_plant, xi_plant, dt)
    plant.analyse()
    omega_ref, xi_ref, omega_err, xi_err = 3., 1., 6., 0.9
    ctl = NLIController(omega_ref, xi_ref, omega_err, xi_err, dt)
    ctl.force_truth(plant)
    time =  np.arange(0., 15.05, dt)
    Yc, Yr = ut.step_vec(time, dt=8), np.zeros((len(time),1))
    U, X = np.zeros((len(time),1)), np.zeros((len(time),2))
    X[0] = [0.5, 0]
    for k in range(1,len(time)):
        Yr[k], U[k-1] = ctl.get(Yc[k-1], X[k-1,0], X[k-2,0], U[k-2,0])
        X[k] = plant.disc_dyn(X[k-1], U[k-1])
        
    so_lti.plot2(time, X, U)
    plt.subplot(2,1,1)
    plt.plot(time, Yc, time, Yr)
    plt.legend(['plant', 'setpoint', 'reference'])
    plt.savefig('../docs/plots/nli__so_lti.png')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
