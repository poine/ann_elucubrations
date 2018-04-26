#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import casadi, mpctools as mpc

import mip_simple, utils as ut, so_lti
import pdb
LOG = logging.getLogger('plant_id__mip_simple')

'''
MPC on MIP simple
'''


class Controller:
    def __init__(self, horizon=50):
        self.horizon = horizon
        filename = "/tmp/plant_id__mip_simple.h5"
        plant_ann = keras.models.load_model(filename)
        #plant_ann.summary()
        w = plant_ann.get_layer(name='plant').get_weights()[0]
        print w
    
        # model
        def dae(x,u):
            _i = np.array([[x[0], x[1], x[2], x[3], u[0]]])
            return np.dot(_i, w)
        #return plant_ann.predict(_i) # doesn't work... :(
        model_dae = mpc.getCasadiFunc(dae, [4, 1], ["x", "u"], funcname="f")

        # stage cost
        scx, scu = np.diag([5000., 100., 10., 1.]), 1.e-6
        def lfunc(x, u, x_sp, u_sp):
            dx = x - x_sp
            return mpc.mtimes(dx.T, scx, dx) + scu*mpc.mtimes(u.T,u)
        stage_cost = mpc.getCasadiFunc(lfunc, [4, 1, 4, 1], ["x", "u", "x_sp", "u_sp"], funcname="l")

        # terminal weight
        tw = np.diag([5000., 100., 10., 1.])
        def Pffunc(x, x_sp):
            dx = x - x_sp
            return mpc.mtimes(dx.T, tw, dx)
        term_weight = mpc.getCasadiFunc(Pffunc, [4, 4], ["x", "x_sp"], funcname="Pf")

        # solver
        u_sat = 10.
        x_m, th_m, xd_m, thd_m = 10, ut.rad_of_deg(40), 3.5, ut.rad_of_deg(400)
        x_bnd = np.array([x_m, th_m, xd_m, thd_m])
        N = {"t":horizon, "x":4, "u":1}
        sp = {'x': np.zeros((horizon+1, 4)), 'u':np.zeros((horizon, 1))}
        args = dict(
            verbosity=0,
            l = stage_cost,
            Pf = term_weight,
            lb={"u" : -u_sat*np.ones((horizon, 1)), 'x': -x_bnd},
            ub={"u" :  u_sat*np.ones((horizon, 1)), 'x': x_bnd},
        )
        self.solver = mpc.nmpc(f=model_dae, N=N, sp=sp, **args)

    def get(self, X, k):
        for i in range(self.horizon+1): # find a way to assign array
            self.solver.par['x_sp',i] =  self.sp[k+i]
        self.solver.fixvar("x", 0, X)
        self.solver.solve()
        u = self.solver.var['u', 0]
        return [u[0]]
        


def main():
    dt = 0.01
    time = np.arange(0, 10.25, dt)
    omega_ref, xi_ref = 5., 0.8
    ref = so_lti.CCPlant(omega_ref, xi_ref)
    X0 = [1, 0, 0, 0]
    _sp = ut.step_vec(time, a0=-0.5, a1=0.5)
    _c = lambda _x, _k: _sp[_k]
    Xref, Uref = ref.sim(time, [X0[0], X0[2]], _c)

    sp = np.zeros((len(time), 4))
    sp[:,0] = Xref[:,0]
    sp[:,2] = Xref[:,1]
    plant = mip_simple.Plant()
    ctl = Controller(horizon=150)
    ctl.sp = sp
    time = time[:-ctl.horizon]
    X, U = plant.sim_with_input_fun(time, ctl, X0)
    mip_simple.plot_all(time, X, U)
    plt.subplot(5,1,1); plt.plot(time, sp[:-ctl.horizon,0])
    plt.subplot(5,1,2); plt.plot(time, sp[:-ctl.horizon,2])
    prefix='test1'
    plt.savefig('../docs/plots/mpc__mip_simple_{}.png'.format(prefix))
    plt.show()

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
