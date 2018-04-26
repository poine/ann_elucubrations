#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Periodic MPC for single-variable system.

# Imports.
import math, numpy as np
import scipy.signal as spsignal
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt
import pdb
import utils as ut

def plot(time, X, U, sp):
    ax = plt.subplot(3, 1, 1)
    plt.plot(time, X[:,0])
    plt.plot(time, sp)
    ut.decorate(ax, title="$\\phi$", ylab='deg')
    ax = plt.subplot(3, 1, 2)
    plt.plot(time, X[:,1])
    ut.decorate(ax, title="$\dot{\phi}$", ylab='deg/s')
    ax = plt.subplot(3, 1, 3)
    plt.plot(time[:-1], U[:,0])
    ut.decorate(ax, title="$\\tau$", ylab='N.m')



# Define optimal periodic solution.
def sp_sawtooth(t, T=1): return spsignal.sawtooth(2*np.pi/T*t + np.pi/2,.5)
def sp_step(t, a0=-1, a1=1, dt=4, t0=0): return np.array([a0 if math.fmod(_t+t0, dt) > dt/2 else a1 for _t in t])

# Define continuous time model.
Acont = np.array([[0, 1], [-1, 0]])
Bcont = np.array([[0], [10]])
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .01
Nt = 500
t = np.arange(Nt + 1)*dt
(Adisc, Bdisc) = mpc.util.c2d(Acont,Bcont,dt)
def F(x, u):
    return mpc.mtimes(Adisc, x) + mpc.mtimes(Bdisc, u)
Fcasadi = mpc.getCasadiFunc(F, [n,m], ["x","u"], funcname="F", scalar=False)


def ode(x,u):
    return np.array([x[1], -10.*np.sin(x[0]) + u[0]])
ode_casadi = mpc.getCasadiFunc(ode, [n, m], ["x", "u"], funcname="f")

# Bounds on u.
umax = 5
lb = {"u" : -umax*np.ones((Nt, m))}
ub = {"u" : umax*np.ones((Nt-1, m))}

# Define Q and R matrices and periodic setpoint.
R = np.eye(m)
Q = np.diag([100, 1])##np.eye(n)
sp = {"x" : sp_step(t)[:,np.newaxis], "u" : np.zeros((Nt, m))}


print sp['x'].shape
print sp['u'].shape
print t.shape, Nt
#pdb.set_trace()

def l(x, u, xsp, usp):
    """Stage cost with setpoints."""
    dx = x - xsp
    du = u - usp
    return mpc.mtimes(dx.T, Q, dx) + mpc.mtimes(du.T, R, du)
lcasadi = mpc.getCasadiFunc(l, [n,m,n,m], ["x","u","x_sp","u_sp"],
                            funcname= "l")

# Initial condition.
x0 = np.array([-2])
N = {"x" : n, "u" : m, "t" : Nt}
funcargs = {"f" : ["x","u"], "l" : ["x","u","x_sp","u_sp"]}

# Solve linear MPC problem.
solution = mpc.callSolver(mpc.nmpc(Fcasadi, lcasadi, N, x0, lb, ub, sp=sp,
                                   funcargs=funcargs, verbosity=3))
x = solution["x"]
u = solution["u"]

# Plot things.
#fig = mpcplots.mpcplot(x, u, t, sp["x"])
#mpcplots.showandsave(fig,"periodicmpcexample.pdf")
plot(t, x, u, sp['x'])
plt.show()
