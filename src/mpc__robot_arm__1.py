#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Model Predictive Control of a robot arm using exact continuous time model
'''

import logging, numpy as np, math, matplotlib.pyplot as plt
import pdb
import utils as ut, robot_arm
import casadi, mpctools as mpc


class NMPController:
    def __init__(self, horizon, sp, plant, u_sat=10.):
        self.horizon, self.sp, self.plant = horizon, sp, plant

        # model
        def ode(x,u): return plant.cont_dyn(x, 0., u)
        model_ode = mpc.getCasadiFunc(ode, [plant.s_size, plant.i_size], ["x", "u"], funcname="f")

        # stage cost
        self.scx, self.scu = np.diag([5000., 100.]), 1.e-3
        def lfunc(x, u, x_sp, u_sp):
            dx = x - x_sp
            return mpc.mtimes(dx.T, self.scx, dx) + self.scu*mpc.mtimes(u.T,u)
        stage_cost = mpc.getCasadiFunc(lfunc, [plant.s_size, plant.i_size, plant.s_size, plant.i_size], ["x", "u", "x_sp", "u_sp"], funcname="l")

        # terminal weight
        self.tw = np.diag([5000., 100.])
        def Pffunc(x, x_sp):
            dx =  x - x_sp
            return mpc.mtimes(dx.T, self.tw, dx)
        term_weight = mpc.getCasadiFunc(Pffunc, [plant.s_size, plant.s_size], ["x", "x_sp"], funcname="Pf")

        # solver
        N = {"t":self.horizon, "x":plant.s_size, "u":plant.i_size, "c":2}
        sp = {'x': np.zeros((self.horizon+1, plant.s_size)), 'u':np.zeros((self.horizon, plant.i_size))}
        args = dict(
            verbosity=0,
            l = stage_cost,
            Pf = term_weight,
            lb={"u" : -u_sat*np.ones((self.horizon, plant.i_size))},
            ub={"u" :  u_sat*np.ones((self.horizon, plant.i_size))},
        )
        self.solver = mpc.nmpc(f=model_ode, N=N, Delta=0.01, sp=sp, **args)
        
    def __call__(self, X, t, k):
        for i in range(self.horizon+1): # find a way to assign array
            self.solver.par['x_sp',i] =  self.sp[k+i]
        #pdb.set_trace()
        self.solver.fixvar("x", 0, X)
        self.solver.solve()
        u = self.solver.var['u', 0]
        return [u[0]]
    


    
def run_sim(time, sp, horizon=50):
   
    X0 = [0.2, 1.]
    plant = robot_arm.Plant()
    ctl = NMPController(horizon, sp, plant)

    time = time[:-ctl.horizon]
    X, U = plant.sim(time, X0, ctl)
    robot_arm.plot(time, X, U, ref=sp[:-ctl.horizon])
    plt.savefig('../docs/images/robot_arm__mpc__1.png')
    plt.show()


def main():
    dt = 0.01
    time = np.arange(0, 5.25, dt)
    
    sp = np.vstack((ut.step_vec(time), np.zeros(len(time)))).T
    #sp = ut.ref_sine_vec(time)
    run_sim(time, sp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
