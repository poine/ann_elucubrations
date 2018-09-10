#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import casadi, mpctools as mpc

import bicycle_dynamics as bd, bicycle_kinematics as bk, utils as ut, two_d_guidance
import casadi, mpctools as mpc
import pdb
LOG = logging.getLogger('mpc__bicycle_kinematics')

'''
MPC on Bicycle_Kinematics
'''


class Controller:
    def __init__(self, path_file, dt=0.01, horizon=50):
        self.path = two_d_guidance.Path(load=path_file)
        self.dt, self.horizon = dt, horizon

        P = bk.Param()
        _nx, _nu = bk.s_size, bk.i_size
        # model
        def ode(x,u): return bk.dyn(x, 0., u, P)
        model_ode = mpc.getCasadiFunc(ode, [_nx, _nu], ["x", "u"], funcname="f")

        
        # stage cost
        #scx, scu = np.diag([5000., 100., 10., 1., 1., 1.]), np.diag([1., 1])
        scx, scu = np.diag([1., 200., 10., 12.]), np.diag([1., 1]) # kinematics
        def lfunc(x, u, x_sp, u_sp):
            dx, du = x - x_sp, u - u_sp
            cost = mpc.mtimes(dx.T, scx, dx) + mpc.mtimes(du.T, scu, du)
            print( cost)
            return cost
        stage_cost = mpc.getCasadiFunc(lfunc, [_nx, _nu, _nx, _nu], ["x", "u", "x_sp", "u_sp"], funcname="l")
  
        # terminal cost
        _tc = np.diag([1., 200., 10., 1.])
        def Pffunc(x, x_sp):
            dx =  x - x_sp
            return mpc.mtimes(dx.T, _tc, dx)
        term_weight = mpc.getCasadiFunc(Pffunc, [_nx, _nx], ["x", "x_sp"], funcname="Pf")

        # solver
        u_sat = np.array([10., ut.rad_of_deg(30.)])
        N = {"t":horizon, "x":_nx, "u":_nu}
        sp = {'x': np.zeros((horizon+1, _nx)), 'u':np.zeros((horizon, _nu))}
        args = dict(
            verbosity=0,
            l = stage_cost,
            Pf = term_weight,
            lb={"u" : -u_sat},
            ub={"u" :  u_sat},
        )
        #pdb.set_trace()
        self.solver = mpc.nmpc(f=model_ode, N=N, Delta=dt, sp=sp, **args)
        
    def compute(self, X, t):
        #for i in range(self.horizon+1): # find a way to assign array
        #    self.solver.par['x_sp',i] =  self.sp[k+i]
        #self.path.find_closest(self, p0, max_look_ahead=100)
        _v_sp = 1.
        for k in range(self.horizon+1):
            self.solver.par['x_sp', k] = [(t+k*self.dt)*_v_sp, 0., 0., _v_sp]
        v = np.linalg.norm(X[bd.s_vx:bd.s_vy+1])
        self.solver.fixvar("x", 0, [X[bd.s_x], X[bd.s_y], X[bd.s_psi], v])
        self.solver.solve()
        #pdb.set_trace()
        #u = self.solver.var['u', 0]
        return self.solver.var['u', 0].full().squeeze()#[u[0], u[1]]
        #return [0., -0.1]

def main():
    _plant = bd.Plant()
    #path_file='/home/poine/work/ann_elucubrations/data/paths/oval_01.npz'
    path_file='/home/poine/work/ann_elucubrations/data/paths/track_ethz_dual_01.npz'
    _ctl = Controller(path_file)
    

    dt = 0.01
    _len = 500
    time = np.linspace(0., _len*dt, _len)
    X, U = np.zeros((_len, bd.s_size)), np.zeros((_len, bd.i_size))
    X0= [0., 0.1, 0.1, 1., 0, 0] #X0= [-.2, 3.2, 0., 0.1, 0, 0]
    X[0] = X0
    for k in range(1, len(time)):
        try:
            U[k-1] = _ctl.compute(X[k-1], time[k-1])
            #pdb.set_trace()
        except two_d_guidance.EndOfPathException:
            print('finished at {}s'.format(time[k]));
            _len=k;  break
        X[k] = _plant.disc_dyn(X[k-1], U[k-1], dt)
    U[_len-1] = U[_len-2]
    bd.plot_time(time, X, U)
    bd.plot2D(time, X, U)
    plt.show()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
