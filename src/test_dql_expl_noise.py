#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import ddpg_utils
import pdb

if __name__ == '__main__':
    ndim=2
    sigma = np.array([0.01, 0.1])
    actor_noise = ddpg_utils.OrnsteinUhlenbeckActionNoise(mu=np.zeros(ndim),
                                                          sigma=sigma)

    samples = np.array([actor_noise() for i in range(10000)])

    if 0:
        plt.figure()
        _ep = np.arange(len(samples))
        _sigmas = sigma*np.exp(-0.003*_ep)
        plt.plot(3*_sigmas)
        plt.plot(-3*_sigmas)

    #pdb.set_trace()
    plt.figure()
    plt.plot(samples)
    plt.plot(3*sigma*np.ones_like(samples))
    plt.plot(-3*sigma*np.ones_like(samples))
    plt.figure()
    for i in range(ndim):
        plt.subplot(1, ndim, i+1)
        plt.hist(samples[:,i])
    plt.show()
