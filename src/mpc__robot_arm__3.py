#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Model Predictive Control of a robot arm using an ANN model model
'''

import logging, numpy as np, math, matplotlib.pyplot as plt
import casadi, mpctools as mpc

import utils as ut, robot_arm, plant_id__robot_arm__fs
import pdb

# it looks like i can not use sklearn perceptron prediction function in casadi... fuck that :(
# let's try to recreate it for now
def test(ann):
    _c, _i = ann.ann.coefs_[0], ann.ann.intercepts_[0]
    def ann_predict(_input):
        _output = np.dot(_input, _c) + _i
        return _output

    def ann_predict2(x, u):
        _input = np.array([[x[0], x[1], u[0]]])
        _output = np.dot(_input, _c) + _i
        return _output[0]
    
    samples = np.random.uniform(low=-1, high=1, size=(10,3))
    for s in samples:
        pred_sk = ann.ann.predict(s.reshape(1,-1))[0]
        #pred_me = ann_predict(s.reshape(1,-1))
        pred_me = ann_predict2([s[0], s[1]], [s[2]])
        print pred_sk, pred_me, pred_sk - pred_me



class NMPController:
    def __init__(self, horizon, sp, plant, u_sat=10., dt=0.01):
        self.horizon, self.sp, self.plant = horizon, sp, plant

        # model
        #Ac, Bc = plant.jacobian([0, 0], [0])
        #(Ad, Bd) = mpc.util.c2d(Ac,Bc,dt)
        #def dae(x,u): return mpc.mtimes(Ad, x) + mpc.mtimes(Bd, u)
        ann = plant_id__robot_arm__fs.ANN_Plant()
        ann_plant_filename = '/tmp/robot_arm__plant_id__fs_ann.pkl'
        ann.load(ann_plant_filename)
        test(ann)
        _c, _i = ann.ann.coefs_[0], ann.ann.intercepts_[0]
        def dae(x, u):
            _input = np.array([[x[0], x[1], u[0]]])
            _output = np.dot(_input, _c) + _i
            return _output
                
        #def dae(x,u): return ann.ann.predict([[x[0], x[1], u[0]]])
        
        model_dae = mpc.getCasadiFunc(dae, [plant.s_size, plant.i_size], ["x", "u"], funcname="f")

        # stage cost
        self.scx, self.scu = np.diag([5000., 100.]), 1.e-3
        def lfunc(x, u, x_sp, u_sp):
            dx = x - x_sp
            return mpc.mtimes(dx.T, self.scx, dx) + self.scu*mpc.mtimes(u.T,u)
        stage_cost = mpc.getCasadiFunc(lfunc, [plant.s_size, plant.i_size, plant.s_size, plant.i_size], ["x", "u", "x_sp", "u_sp"], funcname="l")

        # terminal weight
        self.tw = np.diag([5000., 100.])
        def Pffunc(x, x_sp):
            dx = x - x_sp
            return mpc.mtimes(dx.T, self.tw, dx)
        term_weight = mpc.getCasadiFunc(Pffunc, [plant.s_size, plant.s_size], ["x", "x_sp"], funcname="Pf")

        # solver
        N = {"t":self.horizon, "x":plant.s_size, "u":plant.i_size}
        sp = {'x': np.zeros((self.horizon+1, plant.s_size)), 'u':np.zeros((self.horizon, plant.i_size))}
        args = dict(
            verbosity=0,
            l = stage_cost,
            Pf = term_weight,
            lb={"u" : -u_sat*np.ones((self.horizon, plant.i_size))},
            ub={"u" :  u_sat*np.ones((self.horizon, plant.i_size))},
        )
        self.solver = mpc.nmpc(f=model_dae, N=N, sp=sp, **args)
        
    def __call__(self, X, t, k):
        for i in range(self.horizon+1): # find a way to assign array
            self.solver.par['x_sp',i] =  self.sp[k+i]
        #pdb.set_trace()
        self.solver.fixvar("x", 0, X)
        self.solver.solve()
        u = self.solver.var['u', 0]
        return [u[0]]
    


    
def main():
    dt, horizon = 0.01, 50
    time = np.arange(0, 5.25, dt)
    sp = np.vstack((ut.step_vec(time), np.zeros(len(time)))).T
    #sp = ut.ref_sine_vec(time)
    X0 = [0.2, 1.]
    plant = robot_arm.Plant()
    ctl = NMPController(horizon, sp, plant)

    time = time[:-ctl.horizon]
    X, U = plant.sim(time, X0, ctl)
    robot_arm.plot(time, X, U, ref=sp[:-ctl.horizon])
    plt.savefig('../docs/images/robot_arm__mpc__3.png')
    plt.show()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
