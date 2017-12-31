import logging, math, numpy as np, scipy.integrate, matplotlib.pyplot as plt

import utils as ut

LOG = logging.getLogger('lti_first_order')

class Plant:
    def __init__(self, tau=1., dt=0.01):
        self.tau, self.dt = tau, dt
        self.ad, self.bd = np.exp(-dt/tau), 1. - np.exp(-dt/tau)
        LOG.info('  tau {} dt {}'.format(tau, dt))
        LOG.info('  discrete time: ad {:.5f} bd {:.5f}'.format(self.ad, self.bd))

    def cont_dyn(self, X, t, U):
        Xd =  -1./self.tau*(X-U)
        return Xd

    def disc_dyn(self, Xk, Uk):
        _unused, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [0, self.dt], args=(Uk,))
        return Xkp1

    def disc_dyn2(self, Xk, Uk):
        Xkp1 = self.ad*Xk+self.bd*Uk
        return Xkp1

    def sim(self, time, X0, ctl):
        X, U = np.zeros((len(time), 1)),  np.zeros((len(time), 1))
        X[0] = X0
        for i in range(1, len(time)):
            U[i-1] = ctl(X[i-1], i-1)
            X[i] = self.disc_dyn(X[i-1], U[i-1])
        U[-1] = U[-2]
        return X, U



def plot(time, X, U=None, Yc=None):
    ax = plt.subplot(2,1,1)
    plt.plot(time, X[:,0])
    if Yc is not None: plt.plot(time, Yc, 'k')
    ut.decorate(ax, title="$x_1$", ylab='time')
    if U is not None:
        ax = plt.subplot(3,1,3)
        plt.plot(time, U)
        ut.decorate(ax, title="$u$", ylab='time')
