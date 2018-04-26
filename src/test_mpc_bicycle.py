#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging, numpy as np, matplotlib.pyplot as plt, pickle, os
import keras, control
import casadi, mpctools as mpc

import bicycle_dynamics as bd, bicycle_kinematics as bk, utils as ut, two_d_guidance
import casadi, mpctools as mpc
import pdb


def main():
    _plant = bd.Plant()

    dt, _len = 0.01, 500
    time = np.linspace(0., _len*dt, _len)
    Xsp = np.zeros((_len, bk.s_size))
    _vsp = 0.9
    Xsp[:,bk.s_x] = [k*dt*_vsp for k in range(_len)]
    Xsp[:,bk.s_v] = _vsp

    X, U = np.zeros((_len, bd.s_size)), np.zeros((_len, bd.i_size))

    P = bk.Param()
    _nx, _nu = bk.s_size, bk.i_size
    # model
    def ode(x,u): return bk.dyn(x, 0., u, P)
    model_ode = mpc.getCasadiFunc(ode, [_nx, _nu], ["x", "u"], funcname="f")
    # stage cost
    scx, scu = np.diag([1., 200., 10., 12.]), np.diag([1., 1]) # kinematics
    def lfunc(x, u, x_sp, u_sp):
        dx, du = x - x_sp, u - u_sp
        cost = mpc.mtimes(dx.T, scx, dx) + mpc.mtimes(du.T, scu, du)
        print cost
        return cost
    stage_cost = mpc.getCasadiFunc(lfunc, [_nx, _nu, _nx, _nu], ["x", "u", "x_sp", "u_sp"], funcname="l")
    # terminal cost
    _tc = np.diag([1., 200., 10., 1.])
    def Pffunc(x, x_sp):
        dx =  x - x_sp
        return mpc.mtimes(dx.T, _tc, dx)
    term_weight = mpc.getCasadiFunc(Pffunc, [_nx, _nx], ["x", "x_sp"], funcname="Pf")
    # solver
    horizon=_len-1
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
    solver = mpc.nmpc(f=model_ode, N=N, Delta=dt, sp=sp, **args)
    
    for k in range(horizon+1):
        solver.par['x_sp', k] = Xsp[k]
    solver.fixvar("x", 0, [0., 0.1, 0, 0.9])
    solver.solve()

    X_solver = np.array([solver.var['x', k].full() for k in range(horizon)])
    U_solver = np.array([solver.var['u', k].full() for k in range(horizon)])
    pdb.set_trace()
    
    plots = [("$x$", "m", Xsp[:,bk.s_x]),
             ("$y$", "m", Xsp[:,bk.s_y]),
             ("$\psi$", "m", Xsp[:,bk.s_psi]),
             ("$v$", "m/s", X[:,bk.s_v])
             ]
    for i, (title, ylab, data) in enumerate(plots):
        ax = plt.subplot(6, 1, i+1)
        plt.plot(time, data, linewidth=2)
        ut.decorate(ax, title=title, ylab=ylab)
    plots = [("$x$", "m", X_solver[:,bk.s_x]),
             ("$y$", "m", X_solver[:,bk.s_y]),
             ("$\psi$", "rad", X_solver[:,bk.s_psi]),
             ("$v$", "m/s", X_solver[:,bk.s_v]),
             ("$a$", "m/s", U_solver[:,bk.i_a]),
             ("$df$", "m/s", U_solver[:,bk.i_df])
    ]
    for i, (title, ylab, data) in enumerate(plots):
        ax = plt.subplot(len(plots), 1, i+1)
        plt.plot(time[:-1], data, linewidth=2)
        ut.decorate(ax, title=title, ylab=ylab)  
    plt.show()




if __name__ == "__main__":
    main()
