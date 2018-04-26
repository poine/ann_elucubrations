#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
  Model Predictive Control of a robot arm
'''

import logging, numpy as np, math, scipy.integrate, matplotlib.pyplot as plt

import pdb
import utils as ut, robot_arm
import casadi
import mpctools as mpc
def sp_step(t, a0=-1, a1=1, dt=4, t0=0): return np.array([a0 if math.fmod(_t+t0, dt) > dt/2 else a1 for _t in t])

# Define model
class Plant:
    Nx, Nu = 2, 1

    gol = 1.
    
    
    @staticmethod
    def ode(x,u):
        return np.array([x[1], -Plant.gol*np.sin(x[0]) + u[0]])

    @staticmethod
    def ode2(X, t, U): return Plant.ode(X, U)
    
    @staticmethod
    def disc_dyn(Xk, Uk):
        dt=0.01
        _unused, Xkp1 = scipy.integrate.odeint(Plant.ode2, Xk, [0, dt], args=(Uk,))
        return Xkp1

    @staticmethod
    def disc_dyn2(Xk, Uk):
        dt=0.01
        _unused, Xkp1 = scipy.integrate.odeint(Plant.ode2, Xk, [0, dt], args=(Uk,))
        return Xkp1

    
    @staticmethod
    def sim(X0, time, ctl):
        X = np.zeros((len(time), Plant.Nx))
        U = np.zeros((len(time), Plant.Nu))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl.get(X[i-1], time[i-1], i-1)
            X[i] = Plant.disc_dyn(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U



Acont = np.array([[0, 1], [-Plant.gol, 0]])
Bcont = np.array([[0], [1]])
(Adisc, Bdisc) = mpc.util.c2d(Acont,Bcont,0.01)

def dyn_disc(x,u): return mpc.mtimes(Adisc, x) + mpc.mtimes(Bdisc, u)

def plot(time, X, U, sp=None):
    ax = plt.subplot(3, 1, 1)
    plt.plot(time, ut.deg_of_rad(X[:,0]))
    if sp is not None: plt.plot(time, ut.deg_of_rad(sp[:,0]))
    ut.decorate(ax, title="$\\phi$", ylab='deg')
    ax = plt.subplot(3, 1, 2)
    plt.plot(time, ut.deg_of_rad(X[:,1]))
    ut.decorate(ax, title="$\dot{\phi}$", ylab='deg/s')
    ax = plt.subplot(3, 1, 3)
    plt.plot(time, U[:,0])
    #plt.plot(time, U[:,1])
    ut.decorate(ax, title="$\\tau$", ylab='N.m')


    

class MyNMPController:
    def __init__(self, use_sp=False, use_disc_model=True):
        
        self.horizon = 25

        # model
        if use_disc_model:
            self.model_ode = mpc.getCasadiFunc(dyn_disc, [Plant.Nx, Plant.Nu], ["x", "u"], funcname="f")
        else:
            self.model_ode = mpc.getCasadiFunc(Plant.ode, [Plant.Nx, Plant.Nu], ["x", "u"], funcname="f")

        # stage cost
        self.scx, self.scu = np.diag([5000., 100.]), 1.
        if use_sp:  
            def lfunc(x, u, x_sp, u_sp):
                dx = x - x_sp
                return mpc.mtimes(dx.T, self.scx, dx) + self.scu*mpc.mtimes(u.T,u)
            self.stage_cost = mpc.getCasadiFunc(lfunc, [Plant.Nx, Plant.Nu, Plant.Nx, Plant.Nu], ["x", "u", "x_sp", "u_sp"], funcname="l")
        else:
            def lfunc(x, u): return mpc.mtimes(x.T, self.scx, x) + self.scu*mpc.mtimes(u.T,u)
            self.stage_cost = mpc.getCasadiFunc(lfunc, [Plant.Nx, Plant.Nu], ["x", "u"], funcname="l")
            
        # terminal weight
        self.tw = 50000
        if 0:
            def Pffunc(x, x_sp): return self.tw*mpc.mtimes(x.T,x)
            self.term_weight = mpc.getCasadiFunc(Pffunc, [Plant.Nx, Plant.Nx], ["x", "x_sp"], funcname="Pf")

            
        if use_disc_model:
            N = {"t":self.horizon, "x":Plant.Nx, "u":Plant.Nu}
        else:
            N = {"t":self.horizon, "x":Plant.Nx, "u":Plant.Nu, "c":2}
        sp = dict(
            #x= np.ones((self.horizon+1, Plant.Nx)),
            x=np.vstack((np.ones(self.horizon+1), np.zeros(self.horizon+1))).T, 
            u= np.zeros((self.horizon, Plant.Nu))
        )
        U_sat = 10
        args = dict(
            verbosity=0,
            l=self.stage_cost,
            #Pf=self.term_weight,
            lb={"u" : -U_sat*np.ones((self.horizon, Plant.Nu))},
            ub={"u" :  U_sat*np.ones((self.horizon, Plant.Nu))},
        )
        if use_sp:
            print sp['x'].shape
            print sp['u'].shape
            print self.horizon
            funcargs = {"f" : ["x","u"], "l" : ["x","u","x_sp","u_sp"]}
            self.solver = mpc.nmpc(f=self.model_ode, N=N, Delta=0.1, funcargs=funcargs, sp=sp, **args)
        else:
            funcargs = {"f" : ["x","u"], "l" : ["x","u"]}
            self.solver = mpc.nmpc(f=self.model_ode, N=N, Delta=0.1, **args)
            
        
   
  
    
    def get(self, X, t, k):
        #mpc.util.casadiStruct2numpyDict(self.solver.par)['x_sp'] = self.x_sp[k:k+self.horizon+1]
        for i in range(self.horizon): # find a way to assign array
            self.solver.par['x_sp',i,0] =  self.x_sp[k+i,0]
            self.solver.par['x_sp',i,1] =  self.x_sp[k+i,1]
            
        #pdb.set_trace()
        self.solver.fixvar("x", 0, X)
        self.solver.solve()
        foo = self.solver.var['u', 0]
        return [foo[0]]

        

def main():

    dt = 0.01
    time = np.arange(0, 3.25, dt)
    x_sp = np.vstack((sp_step(time, t0=3.75), np.zeros(len(time)))).T
    time = np.arange(0, 3, dt)
    X0 = [0.5, 1]
    if 0: # use real(aka perfect) plant model
        ctl = MyNMPController(use_disc_model=False)
        X, U = Plant.sim(X0, time, ctl)
        plot(time, X, U)

    if 1: # use linearized model
        ctl = MyNMPController(use_sp=True, use_disc_model=True)
        ctl.x_sp = x_sp
        X, U = Plant.sim(X0, time, ctl)
        plot(time, X, U, x_sp[:300])


    plt.show()
    
  
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    main()
